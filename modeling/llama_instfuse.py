
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

@dataclass
class CausalLMFuseOutputWithPast(CausalLMOutputWithPast):
    past_inst_hidden_states: Optional[torch.FloatTensor] = None

class LlamaForCausalLMFuse(transformers.LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.fuse_head = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.post_init()

    def _compute_shifts(self, expert_labels: torch.LongTensor) -> Tuple[torch.LongTensor, torch.BoolTensor]:
        batch_size, seq_len = expert_labels.shape

        # Find the last zero index for each sequence
        zero_mask = (expert_labels == 0)  # (batch_size, seq_len)

        # Get the last index where zero occurs
        last_zero_indices = (zero_mask * torch.arange(seq_len, device=expert_labels.device)).max(dim=1)[0] # (batch_size, )

        # If no zero exists in the sequence, set last_zero_indices to seq_len - 1
        no_zero_mask = ~zero_mask.any(dim=1)  # Rows where no zero exists
        last_zero_indices = torch.where(no_zero_mask, torch.tensor(seq_len - 1, device=expert_labels.device), last_zero_indices) # (batch_size, )

        # Generate mask for indices > last_zero_indices
        orig_indices = torch.arange(seq_len, device=expert_labels.device).expand(batch_size, -1)  # (batch_size, seq_len)
        mask = orig_indices > last_zero_indices.unsqueeze(1)  # (batch_size, seq_len)

        return last_zero_indices, mask

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
            batch_size, _, hidden_size = hidden_states.shape
            if past_inst_hidden_states is None:
                last_zero_indices, mask = self._compute_shifts(expert_labels)

                last_zero_indices = last_zero_indices.unsqueeze(1).unsqueeze(2).expand(-1, 1, hidden_size)  # (batch_size, 1, cut_hidden_size)
                past_inst_hidden_states = torch.gather(hidden_states, dim=1, index=last_zero_indices)  # (batch_size, 1, cut_hidden_size) cache_last_inst_hidden[i][j][k] = hidden_states[i][last_zero_indices[i]][k]

                # Add instruction semantic as a residual connection
                mask = mask.unsqueeze(-1).expand(-1, -1, hidden_size)  # (batch_size, seq_len, hidden_size)
                hidden_states = hidden_states + mask * past_inst_hidden_states
            else:
                # hidden_states = hidden_states + past_inst_hidden_states
                pass

            if self.config.pretraining_tp > 1:
                fuse_head_slices = self.fuse_head.weight.split(self.hidden_size // self.config.pretraining_tp, dim=0)
                hidden_states = [F.linear(hidden_states, fuse_head_slices[i]) for i in range(self.config.pretraining_tp)]
                hidden_states = torch.cat(hidden_states, dim=-1)
            else:
                hidden_states = self.fuse_head(hidden_states) # additional fusion layer

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
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
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values,
            attention_mask,
            inputs_embeds,
            cache_position,
            position_ids,
            use_cache,
            **kwargs,
        )

        if "expert_labels" in kwargs:
            model_inputs["expert_labels"] = kwargs["expert_labels"]  # Pass expert_labels if provided
        if "past_inst_hidden_states" in kwargs:
            model_inputs["past_inst_hidden_states"] = kwargs["past_inst_hidden_states"]  # Pass expert_labels if provided
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
        if "past_inst_hidden_states" in outputs:
            model_kwargs["past_inst_hidden_states"] = outputs.past_inst_hidden_states.detach() # cache the last instruction token hidden state
        return model_kwargs
