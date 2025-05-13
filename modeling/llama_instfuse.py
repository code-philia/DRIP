
import transformers
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import ACT2FN
from transformers.utils import logging
from typing import Union, Optional, Tuple, List, Dict, Any
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast, ModelOutput
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache, DynamicCache
import torch.nn.functional as F
from functools import partial
from dataclasses import dataclass
logger = logging.get_logger(__name__)

class LlamaFuseConfig(LlamaConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_dict(self):
        """Ensures the custom keys are included when saving config."""
        base_dict = super().to_dict()
        return base_dict


@dataclass
class CausalLMFuseOutputWithPast(CausalLMOutputWithPast):
    past_inst_hidden_states: Optional[torch.FloatTensor] = None

class LlamaForCausalLMFuse(transformers.LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: LlamaFuseConfig):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        self.label_gap = nn.Parameter(torch.tensor(0.05))
        self.post_init()

    def _get_resp_indices(self, expert_labels: torch.LongTensor) -> torch.LongTensor:
        batch_size, seq_len = expert_labels.shape
        # mask of where label == 2
        two_mask = (expert_labels == 2)  # (B, L)

        # broadcasted sequence indices 0…L−1
        seq_indices = torch.arange(seq_len, device=expert_labels.device).unsqueeze(0)  # (1, L)

        # set “no-2” positions to a big index (= seq_len)
        masked_indices = torch.where(two_mask, seq_indices, seq_len)  # (B, L)

        # take the minimum along dim=1 → shape (B,)
        first_two_indices = masked_indices.min(dim=1)[0]  # (batch_size,)

        return first_two_indices

    def _get_inst_indices(self, expert_labels: torch.LongTensor) -> torch.LongTensor:
        batch_size, seq_len = expert_labels.shape

        # Find the last zero index for each sequence
        zero_mask = (expert_labels == 0)  # (batch_size, seq_len)

        # Get the last index where zero occurs
        last_zero_indices = (zero_mask * torch.arange(seq_len, device=expert_labels.device)).max(dim=1)[0] # (batch_size, )

        # If no zero exists in the sequence, set last_zero_indices to seq_len - 1
        no_zero_mask = ~zero_mask.any(dim=1)  # Rows where no zero exists
        last_zero_indices = torch.where(no_zero_mask, torch.tensor(seq_len - 1, device=expert_labels.device), last_zero_indices) # (batch_size, )

        return last_zero_indices

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        expert_labels: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        past_inst_hidden_states: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMFuseOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0] # (batch_size, seq_len, hidden_size)
        if expert_labels is not None:
            batch_size, length, hidden_size = hidden_states.shape
            if past_inst_hidden_states is None:
                batch_idx = torch.arange(batch_size, device=hidden_states.device) # (B,)

                last_inst_indices = self._get_inst_indices(expert_labels) # (B,)
                last_inst = hidden_states[batch_idx, last_inst_indices]  # (B,H)

                first_resp_indices = self._get_resp_indices(expert_labels) # (B,)
                first_resp = hidden_states[batch_idx, first_resp_indices]  # (B,H)

                # fixme: Add last instruction token's semantic as a residual connection to the 1st response token
                mix_factor = torch.sigmoid(self.residual_weight)  # factor between 0 and 1
                mixed_states = torch.lerp(first_resp, last_inst, mix_factor) # (B,H)

                mask2d = torch.zeros(batch_size, length, dtype=torch.bool, device=hidden_states.device)
                mask2d[batch_idx, first_resp_indices] = True
                mask3d = mask2d.unsqueeze(-1) # broadcast to (B, L, 1)

                mixed = mixed_states.unsqueeze(1).expand(-1, length, -1) # expand mixed_states (B,H) → (B, L, H)
                hidden_states = torch.where(mask3d, mixed, hidden_states)

                # fixme: Reserve a dimension for the instruction hierarchy level
                tags = expert_labels.to(hidden_states.dtype) * self.label_gap # (B,L)
                new_hidden = torch.cat([
                    tags.unsqueeze(-1),  # (B, L, 1)
                    hidden_states[..., 1:]  # (B, L, H-1)
                ], dim=-1)  # => (B, L, H)
                hidden_states = new_hidden

                past_inst_hidden_states = last_inst.clone().detach().cpu()

            else: # for newly generated token, do not add the instruction semantic
                tags = expert_labels.to(hidden_states.dtype) * self.label_gap  # (B,L)
                new_hidden = torch.cat([
                    tags.unsqueeze(-1),  # (B, L, 1)
                    hidden_states[..., 1:]  # (B, L, H-1)
                ], dim=-1)  # => (B, L, H)
                hidden_states = new_hidden

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            hidden_states = hidden_states.to(self.lm_head.weight.dtype)
            logits = self.lm_head(hidden_states)

        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMFuseOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            past_inst_hidden_states=past_inst_hidden_states,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        expert_labels = None
        if "expert_labels" in kwargs:
            expert_labels = kwargs["expert_labels"]

        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
                if expert_labels is not None:
                    expert_labels = expert_labels[:, -cache_position.shape[0] :] # ensure correct slicing
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position] # (batch_size, 1)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
                if expert_labels is not None:
                    expert_labels = expert_labels[:, -input_ids.shape[1] :] # fixme: the expert label will inherit the last token's expert label

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

        if expert_labels is not None:
            model_inputs["expert_labels"] = expert_labels
        if "past_inst_hidden_states" in kwargs:
            model_inputs["past_inst_hidden_states"] = kwargs["past_inst_hidden_states"]  # Pass expert_labels if provided

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:

        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder,
            standardize_cache_format,
            num_new_tokens,
        )
        if hasattr(outputs, "past_inst_hidden_states"):
            model_kwargs["past_inst_hidden_states"] = outputs.past_inst_hidden_states # cache the last instruction token hidden state
        return model_kwargs
