import copy
import gc
import logging
import queue
import threading

from dataclasses import dataclass
from tqdm import tqdm
from typing import List, Optional, Tuple, Union

import torch
import transformers
from torch import Tensor
from transformers import set_seed
from scipy.stats import spearmanr

from gcg.utils import (
    INIT_CHARS,
    configure_pad_token,
    find_executable_batch_size,
    get_nonascii_toks,
    mellowmax,
)
from config import FILTERED_TOKENS, DELIMITERS
from data_generation.struq import _tokenize_fn

logger = logging.getLogger("nanogcg")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class ProbeSamplingConfig:
    draft_model: transformers.PreTrainedModel
    draft_tokenizer: transformers.PreTrainedTokenizer
    r: int = 8
    sampling_factor: int = 16


@dataclass
class GCGConfig:
    num_steps: int = 250
    optim_str_init: Union[str, List[str]] = "x x x x x x x x x x x x x x x x x x x x"
    search_width: int = 512
    batch_size: int = None
    topk: int = 256
    n_replace: int = 1
    buffer_size: int = 0
    use_mellowmax: bool = False
    mellowmax_alpha: float = 1.0
    log_freq: int = 50
    early_stop: bool = True
    use_prefix_cache: bool = True
    allow_non_ascii: bool = False
    filter_ids: bool = True
    add_space_before_target: bool = False
    seed: int = None
    verbosity: str = "INFO"
    probe_sampling_config: Optional[ProbeSamplingConfig] = None


@dataclass
class GCGResult:
    best_loss: float
    best_string: str
    losses: List[float]
    strings: List[str]


class AttackBuffer:
    def __init__(self, size: int):
        self.buffer = []  # elements are (loss: float, optim_ids: Tensor)
        self.size = size

    def add(self, loss: float, optim_ids: Tensor, optim_gen: Optional[str] = None) -> None:
        if self.size == 0:
            self.buffer = [(loss, optim_ids, None)]
            return

        if len(self.buffer) < self.size:
            self.buffer.append((loss, optim_ids, optim_gen))
        else:
            self.buffer[-1] = (loss, optim_ids, optim_gen)

        self.buffer.sort(key=lambda x: x[0])


    def get_best_ids(self) -> Tensor:
        return self.buffer[0][1]

    def get_lowest_loss(self) -> float:
        return self.buffer[0][0]

    def get_highest_loss(self) -> float:
        return self.buffer[-1][0]

    def get_best_gen(self) -> Optional[str]:
        return self.buffer[0][2]

    def log_buffer(self, tokenizer):
        message = "buffer:"
        for loss, ids, _ in self.buffer:
            optim_str = tokenizer.batch_decode(ids)[0]
            optim_str = optim_str.replace("\\", "\\\\")
            optim_str = optim_str.replace("\n", "\\n")
            message += f"\nloss: {loss}" + f" | string: {optim_str}"
        logger.info(message)


def sample_ids_from_grad(
    ids: Tensor,
    grad: Tensor,
    search_width: int,
    topk: int = 256,
    n_replace: int = 1,
    not_allowed_ids: Tensor = False,
):
    """Returns `search_width` combinations of token ids based on the token gradient.

    Args:
        ids : Tensor, shape = (n_optim_ids)
            the sequence of token ids that are being optimized
        grad : Tensor, shape = (n_optim_ids, vocab_size)
            the gradient of the GCG loss computed with respect to the one-hot token embeddings
        search_width : int
            the number of candidate sequences to return
        topk : int
            the topk to be used when sampling from the gradient
        n_replace : int
            the number of token positions to update per sequence
        not_allowed_ids : Tensor, shape = (n_ids)
            the token ids that should not be used in optimization

    Returns:
        sampled_ids : Tensor, shape = (search_width, n_optim_ids)
            sampled token ids
    """
    n_optim_tokens = len(ids)
    original_ids = ids.repeat(search_width, 1)

    if not_allowed_ids is not None:
        grad[:, not_allowed_ids.to(grad.device)] = float("inf") # (n_optim_ids, vocab_size)

    topk_ids = (-grad).topk(topk, dim=1).indices # (n_optim_ids, topk tokens) 每行是该位置最有希望“替换成”哪些 token

    sampled_ids_pos = torch.argsort(torch.rand((search_width, n_optim_tokens), device=grad.device))[..., :n_replace] # search_width个序列，每个序列都是随机选 n_replace 个位置要“换”
    sampled_ids_val = torch.gather(
        topk_ids[sampled_ids_pos],
        2,
        torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device),
    ).squeeze(2) # 在它位置的 topk 候选里再随机挑一个具体的 token

    new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)

    return new_ids


def filter_ids(ids: Tensor, tokenizer: transformers.PreTrainedTokenizer):
    """Filters out sequeneces of token ids that change after retokenization.

    Args:
        ids : Tensor, shape = (search_width, n_optim_ids)
            token ids
        tokenizer : ~transformers.PreTrainedTokenizer
            the model's tokenizer

    Returns:
        filtered_ids : Tensor, shape = (new_search_width, n_optim_ids)
            all token ids that are the same after retokenization
    """
    ids_decoded = tokenizer.batch_decode(ids)
    filtered_ids = []

    for i in range(len(ids_decoded)):
        # Retokenize the decoded token ids
        ids_encoded = tokenizer(ids_decoded[i], return_tensors="pt", add_special_tokens=False).to(ids.device)["input_ids"][0]
        if torch.equal(ids[i], ids_encoded):
            filtered_ids.append(ids[i])

    if not filtered_ids:
        # This occurs in some cases, e.g. using the Llama-3 tokenizer with a bad initialization
        raise RuntimeError(
            "No token sequences are the same after decoding and re-encoding. "
            "Consider setting `filter_ids=False` or trying a different `optim_str_init`"
        )

    return torch.stack(filtered_ids)


class GCG:
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        config: GCGConfig,
        pass_expert_labels: bool,
        frontend_delimiter: str
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        self.embedding_layer = model.get_input_embeddings()
        not_allowed_nonascii = get_nonascii_toks(tokenizer, device=model.device)
        not_allowed_delimiter_ids = torch.tensor([tokenizer(x, add_special_tokens=False).input_ids[0] for x in FILTERED_TOKENS], device=model.device)
        self.not_allowed_ids = torch.cat([not_allowed_nonascii, not_allowed_delimiter_ids], dim=0)
        self.prefix_cache = None

        self.stop_flag = False
        self.pass_expert_labels = pass_expert_labels
        self.frontend_delimiter = frontend_delimiter
        self.log_freq = self.config.log_freq

        if model.dtype in (torch.float32, torch.float64):
            logger.warning(f"Model is in {model.dtype}. Use a lower precision data type, if possible, for much faster optimization.")

        if model.device == torch.device("cpu"):
            logger.warning("Model is on the CPU. Use a hardware accelerator for faster optimization.")

        if not tokenizer.chat_template:
            logger.warning("Tokenizer does not have a chat template. Assuming base model and setting chat template to empty.")
            tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"

    def run(
        self,
        messages: str,
        target: str,
    ) -> GCGResult:

        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        if config.seed is not None:
            set_seed(config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

        response_delimiter = DELIMITERS[self.frontend_delimiter][2]
        if response_delimiter in messages:
            template = messages.split(response_delimiter)[0].strip() + " {optim_str}" + "\n\n" + response_delimiter + "\n"
        else:
            template = messages.strip() + " {optim_str}" + "\n\n" + response_delimiter + "\n"

        # Remove the BOS token -- this will get added when tokenizing, if necessary
        if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
            template = template.replace(tokenizer.bos_token, "")
        before_str, after_str = template.split("{optim_str}\n\n")

        target = " " + target if config.add_space_before_target else target
        target = target + tokenizer.eos_token

        # Tokenize everything that doesn't get optimized
        examples_tokenized   = _tokenize_fn([before_str], tokenizer, frontend_delimiters=self.frontend_delimiter, compute_gate=True)
        before_ids           = examples_tokenized["input_ids"][0].unsqueeze(0).to(model.device, torch.int64)
        before_expert_labels = examples_tokenized["expert_labels"][0].unsqueeze(0).to(model.device, torch.int64)

        examples_tokenized  = _tokenize_fn([after_str], tokenizer, frontend_delimiters=self.frontend_delimiter, compute_gate=True, add_special_tokens=False)
        after_ids           = examples_tokenized["input_ids"][0].unsqueeze(0).to(model.device, torch.int64)
        after_expert_labels = examples_tokenized["expert_labels"][0].unsqueeze(0).to(model.device, torch.int64)

        examples_tokenized = _tokenize_fn([target], tokenizer, frontend_delimiters=self.frontend_delimiter, compute_gate=False, add_special_tokens=False)
        target_ids           = examples_tokenized["input_ids"][0].unsqueeze(0).to(model.device, torch.int64)
        target_expert_labels = 2 * torch.ones_like(target_ids).to(model.device, torch.int64)

        # Embed everything that doesn't get optimized
        embedding_layer = self.embedding_layer
        before_embeds, after_embeds, target_embeds = [embedding_layer(ids) for ids in (before_ids, after_ids, target_ids)]

        # Compute the KV Cache for tokens that appear before the optimized tokens
        if config.use_prefix_cache:
            with torch.no_grad():
                if not self.pass_expert_labels:
                    output = model(inputs_embeds=before_embeds, use_cache=True)
                    self.prefix_cache = output.past_key_values

        self.target_ids = target_ids
        self.before_embeds = before_embeds
        self.after_embeds = after_embeds
        self.target_embeds = target_embeds
        self.before_expert_labels = before_expert_labels
        self.after_expert_labels = after_expert_labels
        self.target_expert_labels = target_expert_labels

        buffer    = self.init_buffer()
        optim_ids = buffer.get_best_ids()

        losses = []
        optim_strings = []

        for step in tqdm(range(config.num_steps)):

            # Compute the token gradient
            optim_ids_onehot_grad = self.compute_token_gradient(optim_ids)

            with torch.no_grad():
                # Sample candidate token sequences based on the token gradient
                sampled_ids = sample_ids_from_grad(
                    optim_ids.squeeze(0),
                    optim_ids_onehot_grad.squeeze(0),
                    config.search_width,
                    config.topk,
                    config.n_replace,
                    not_allowed_ids=self.not_allowed_ids,
                )

                if config.filter_ids: # filter unallowed ids
                    sampled_ids = filter_ids(sampled_ids, tokenizer) # (new_search_width, n_optim_ids)

                new_search_width = sampled_ids.shape[0]

                # Compute loss on all candidate sequences
                batch_size = new_search_width if config.batch_size is None else config.batch_size
                if self.prefix_cache:
                    input_embeds = torch.cat([
                        embedding_layer(sampled_ids),
                        after_embeds.repeat(new_search_width, 1, 1),
                        target_embeds.repeat(new_search_width, 1, 1),
                    ], dim=1)
                    loss = find_executable_batch_size(self._compute_candidates_loss_original, batch_size)(input_embeds)
                else:
                    input_embeds = torch.cat([
                        before_embeds.repeat(new_search_width, 1, 1),
                        embedding_layer(sampled_ids),
                        after_embeds.repeat(new_search_width, 1, 1),
                        target_embeds.repeat(new_search_width, 1, 1),
                    ], dim=1)

                    if self.pass_expert_labels:
                        input_expert_labels = torch.cat(
                            [
                                self.before_expert_labels.repeat(new_search_width, 1), # (new_search_width, L_before_str)
                                torch.ones_like(sampled_ids).to(sampled_ids.device), # (new_search_width, n_optim_ids)
                                self.after_expert_labels.repeat(new_search_width, 1),
                                self.target_expert_labels.repeat(new_search_width, 1),
                            ],
                            dim=1,
                        )
                        loss = find_executable_batch_size(self._compute_candidates_loss_original, batch_size)(input_embeds, input_expert_labels)
                    else:
                        loss = find_executable_batch_size(self._compute_candidates_loss_original, batch_size)(input_embeds)

                current_loss = loss.min().item()
                optim_ids = sampled_ids[loss.argmin()].unsqueeze(0)
                # Update the buffer based on the loss
                losses.append(current_loss)

            if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                buffer.add(current_loss, optim_ids)

            optim_ids = buffer.get_best_ids()
            optim_str = tokenizer.batch_decode(optim_ids)[0]
            optim_strings.append(optim_str)

            # if self.config.early_stop:
            #     full_optim_str = template.replace("{optim_str}", optim_str)
            #     full_tokenized = _tokenize_fn([full_optim_str], tokenizer,
            #                                   frontend_delimiters=self.frontend_delimiter, compute_gate=True,
            #                                   add_special_tokens=False)
            #     full_ids = full_tokenized["input_ids"][0].unsqueeze(0).to(model.device, torch.int64)
            #     full_expert_labels = full_tokenized["expert_labels"][0].unsqueeze(0).to(model.device, torch.int64)
            #     generated_str = self.inference_check(
            #         input_ids=full_ids,
            #         input_expert_labels=full_expert_labels
            #     )
            #     if generated_str.lower().startswith(target.lower()):
            #         self.stop_flag = True
            #     logger.info(f"optim_str: {optim_str}, generated_str: {generated_str}")
            if self.stop_flag:
                logger.info("Early stopping due to finding a perfect match.")
                break

        min_loss_index = losses.index(min(losses))

        result = GCGResult(
            best_loss=losses[min_loss_index],
            best_string=optim_strings[min_loss_index],
            losses=losses,
            strings=optim_strings,
        )

        return result

    def init_buffer(self) -> AttackBuffer:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        logger.info(f"Initializing attack buffer of size {config.buffer_size}...")

        # Create the attack buffer and initialize the buffer ids
        buffer = AttackBuffer(config.buffer_size)

        if isinstance(config.optim_str_init, str):
            init_optim_ids = tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
            if config.buffer_size > 1:
                init_buffer_ids = tokenizer(INIT_CHARS, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze().to(model.device)
                init_indices = torch.randint(0, init_buffer_ids.shape[0], (config.buffer_size - 1, init_optim_ids.shape[1]))
                init_buffer_ids = torch.cat([init_optim_ids, init_buffer_ids[init_indices]], dim=0)
            else:
                init_buffer_ids = init_optim_ids

        else:  # assume list
            if len(config.optim_str_init) != config.buffer_size:
                logger.warning(f"Using {len(config.optim_str_init)} initializations but buffer size is set to {config.buffer_size}")
            try:
                init_buffer_ids = tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
            except ValueError:
                logger.error("Unable to create buffer. Ensure that all initializations tokenize to the same length.")

        true_buffer_size = max(1, config.buffer_size)

        # Compute the loss on the initial buffer entries
        if self.prefix_cache:
            init_buffer_embeds = torch.cat([
                self.embedding_layer(init_buffer_ids),
                self.after_embeds.repeat(true_buffer_size, 1, 1),
                self.target_embeds.repeat(true_buffer_size, 1, 1),
            ], dim=1)
            init_buffer_losses = find_executable_batch_size(self._compute_candidates_loss_original, true_buffer_size)(init_buffer_embeds)

        else:
            init_buffer_embeds = torch.cat([
                self.before_embeds.repeat(true_buffer_size, 1, 1),
                self.embedding_layer(init_buffer_ids),
                self.after_embeds.repeat(true_buffer_size, 1, 1),
                self.target_embeds.repeat(true_buffer_size, 1, 1),
            ], dim=1)
            if self.pass_expert_labels:
                input_expert_labels = torch.cat(
                    [
                        self.before_expert_labels.repeat(true_buffer_size, 1),
                        torch.ones_like(init_buffer_ids).to(init_buffer_ids.device),
                        self.after_expert_labels.repeat(true_buffer_size, 1),
                        self.target_expert_labels.repeat(true_buffer_size, 1),
                    ],
                    dim=1,
                )
                init_buffer_losses = find_executable_batch_size(self._compute_candidates_loss_original, true_buffer_size)(init_buffer_embeds, input_expert_labels)
            else:
                init_buffer_losses = find_executable_batch_size(self._compute_candidates_loss_original, true_buffer_size)(init_buffer_embeds)

        # Populate the buffer
        for i in range(true_buffer_size):
            buffer.add(init_buffer_losses[i], init_buffer_ids[[i]])

        buffer.log_buffer(tokenizer)

        logger.info("Initialized attack buffer.")

        return buffer

    def compute_token_gradient(
        self,
        optim_ids: Tensor,
    ) -> Tensor:
        """Computes the gradient of the GCG loss w.r.t the one-hot token matrix.

        Args:
            optim_ids : Tensor, shape = (1, n_optim_ids)
                the sequence of token ids that are being optimized
        """
        model = self.model
        embedding_layer = self.embedding_layer

        # Create the one-hot encoding matrix of our optimized token ids
        optim_ids_onehot = torch.nn.functional.one_hot(optim_ids, num_classes=embedding_layer.num_embeddings) # (1, n_optim_ids, vocab_size)
        optim_ids_onehot = optim_ids_onehot.to(model.device, model.dtype)
        optim_ids_onehot.requires_grad_()

        # (1, num_optim_tokens, vocab_size) @ (vocab_size, embed_dim) -> (1, num_optim_tokens, embed_dim)
        optim_embeds = optim_ids_onehot @ embedding_layer.weight

        if self.prefix_cache:
            input_embeds = torch.cat([optim_embeds,
                                      self.after_embeds,
                                      self.target_embeds],
                                     dim=1)
            output = model(
                inputs_embeds=input_embeds,
                past_key_values=self.prefix_cache,
                use_cache=True,
            )
        else:
            input_embeds = torch.cat(
                [
                    self.before_embeds,
                    optim_embeds,
                    self.after_embeds,
                    self.target_embeds,
                ],
                dim=1,
            )
            if self.pass_expert_labels:
                input_expert_labels = torch.cat(
                    [
                        self.before_expert_labels,
                        torch.ones_like(optim_ids).to(optim_embeds.device),
                        self.after_expert_labels,
                        self.target_expert_labels,
                    ],
                    dim=1,
                )
                output = model(inputs_embeds=input_embeds,
                               expert_labels=input_expert_labels)
            else:
                output = model(inputs_embeds=input_embeds)

        logits = output.logits # (1, L)

        # Shift logits so token n-1 predicts token n
        shift = input_embeds.shape[1] - self.target_ids.shape[1]
        shift_logits = logits[..., shift - 1 : -1, :].contiguous()  # (1, num_target_ids, vocab_size)
        shift_labels = self.target_ids

        if self.config.use_mellowmax:
            label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
            loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
        else:
            loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        optim_ids_onehot_grad = torch.autograd.grad(outputs=[loss], inputs=[optim_ids_onehot])[0]

        return optim_ids_onehot_grad

    @torch.inference_mode()
    def inference_check(self,
        input_ids: Tensor,
        input_expert_labels: Optional[Tensor] = None,
        max_new_tokens: int =5,
    ) -> str:

        if self.pass_expert_labels:
            output_ids = self.model.generate(
                input_ids,
                expert_labels=input_expert_labels,
                attention_mask=torch.ones_like(input_ids),
                pad_token_id=self.tokenizer.pad_token_id,
                temperature=0,
                do_sample=False,
                max_new_tokens=max_new_tokens
            )
        else:
            output_ids = self.model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                pad_token_id=self.tokenizer.pad_token_id,
                temperature=0,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )

        eos_token_id = self.tokenizer.eos_token_id
        eos_index = (output_ids == eos_token_id).nonzero(as_tuple=True)[1]
        if eos_index.numel() > 0:
            eos_index = eos_index[0].item()  # Get the scalar value of the index
        else:
            eos_index = output_ids.shape[1]  # No EOS token, use the full length
        outp = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:eos_index].tolist(), skip_special_tokens=True)
        del output_ids
        gc.collect()
        torch.cuda.empty_cache()
        return outp

    def _compute_candidates_loss_original(
        self,
        search_batch_size: int,
        input_embeds: Tensor,
        input_expert_labels: Optional[Tensor] = None
    ) -> Tensor:
        """Computes the GCG loss on all candidate token id sequences.

        Args:
            search_batch_size : int
                the number of candidate sequences to evaluate in a given batch
            input_embeds : Tensor, shape = (search_width, seq_len, embd_dim)
                the embeddings of the `search_width` candidate sequences to evaluate
        """
        all_loss = []
        prefix_cache_batch = []

        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i:i + search_batch_size]
                current_batch_size = input_embeds_batch.shape[0]

                if self.prefix_cache:
                    if not prefix_cache_batch or current_batch_size != search_batch_size:
                        prefix_cache_batch = [[x.expand(current_batch_size, -1, -1, -1) for x in self.prefix_cache[i]] for i in range(len(self.prefix_cache))]

                    outputs = self.model(inputs_embeds=input_embeds_batch,
                                         past_key_values=prefix_cache_batch,
                                         use_cache=True)
                else:
                    if self.pass_expert_labels:
                        expert_labels_batch = input_expert_labels[i:i + search_batch_size]
                        outputs = self.model(inputs_embeds=input_embeds_batch,
                                             expert_labels=expert_labels_batch)
                    else:
                        outputs = self.model(inputs_embeds=input_embeds_batch)

                logits = outputs.logits

                tmp = input_embeds.shape[1] - self.target_ids.shape[1] # start pos of target
                shift_logits = logits[..., tmp-1:-1, :].contiguous() # (B, num_target, vocab_size) because the model is doing next-token-prediction
                shift_labels = self.target_ids.repeat(current_batch_size, 1) # (B, num_target)

                if self.config.use_mellowmax:
                    label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
                    loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
                else:
                    loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none")

                loss = loss.view(current_batch_size, -1).mean(dim=-1)
                all_loss.append(loss)

                if self.config.early_stop:
                    if torch.any(torch.all(torch.argmax(shift_logits, dim=-1) == shift_labels, dim=-1)).item():
                        self.stop_flag = True

                del outputs
                gc.collect()
                torch.cuda.empty_cache()

        return torch.cat(all_loss, dim=0)

# A wrapper around the GCG `run` method that provides a simple API
def run(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    messages: Union[str, List[dict]],
    target: str,
    frontend_delimiter: str,
    config: Optional[GCGConfig] = None,
    pass_expert_labels: bool = False,
) -> GCGResult:
    """Generates a single optimized string using GCG.

    Args:
        model: The model to use for optimization.
        tokenizer: The model's tokenizer.
        messages: The conversation to use for optimization.
        target: The target generation.
        config: The GCG configuration to use.

    Returns:
        A GCGResult object that contains losses and the optimized strings.
    """
    if config is None:
        config = GCGConfig()

    logger.setLevel(getattr(logging, config.verbosity))

    gcg = GCG(model, tokenizer, config, pass_expert_labels,
              frontend_delimiter
    )
    result = gcg.run(messages, target)
    return result