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
from .gcg import GCG, GCGConfig, AttackBuffer, GCGResult, ProbeSamplingConfig, logger, sample_ids_from_grad, filter_ids


@dataclass
class NeuralExecConfig(GCGConfig):
    num_steps: int = 250
    optim_str_init_before: str = "x x x x x"
    optim_str_init_after: str = "x x x x x x x x x x x x x x x"
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


class NeuralExecAttackBuffer(AttackBuffer):
    def __init__(self, size: int):
        super().__init__(size)

    def add(self, loss: float,
            optim_ids_before:
            Tensor, optim_ids_after: Tensor,
            optim_gen: Optional[str] = None) -> None:
        if self.size == 0:
            self.buffer = [(loss, optim_ids_before, optim_ids_after, None)]
            return

        if len(self.buffer) < self.size:
            self.buffer.append((loss, optim_ids_before, optim_ids_after, optim_gen))
        else:
            self.buffer[-1] = (loss, optim_ids_before, optim_ids_after, optim_gen)

        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self) -> Tuple[Tensor, Tensor]:
        return (self.buffer[0][1], self.buffer[0][2])

    def log_buffer(self, tokenizer):
        message = "buffer:"
        for loss, ids_before, ids_after, _ in self.buffer:
            optim_str = tokenizer.batch_decode(ids_before)[0]
            optim_str = optim_str.replace("\\", "\\\\")
            optim_str = optim_str.replace("\n", "\\n")

            optim_str_after = tokenizer.batch_decode(ids_after)[0]
            optim_str_after = optim_str_after.replace("\\", "\\\\")
            optim_str_after = optim_str_after.replace("\n", "\\n")
            message += f"\nloss: {loss}" + f" | prefix: {optim_str} | suffix: {optim_str_after}"
        logger.info(message)


@dataclass
class NeuralExecResult(GCGResult):
    best_loss: float
    best_string: Tuple[str, str]
    losses: List[float]
    strings: List[Tuple[str, str]]

class NeuralExec(GCG):
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        config: NeuralExecConfig,
        pass_expert_labels: bool,
        frontend_delimiter: str
    ):
        super().__init__(model, tokenizer, config, pass_expert_labels, frontend_delimiter)

    def run(
        self,
        messages: str,
        enclosed_instruction: str,
        target: str,
    ) -> NeuralExecResult:

        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        if config.seed is not None:
            set_seed(config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

        response_delimiter = DELIMITERS[self.frontend_delimiter][2]
        if response_delimiter in messages:
            template = messages.split(response_delimiter)[0].strip() + " {optim_str_before} " + enclosed_instruction + " {optim_str_after} \n\n" + response_delimiter + "\n"
        else:
            template = messages.strip()  + " {optim_str_before} " + enclosed_instruction + " {optim_str_after} \n\n" + response_delimiter + "\n"

        # Remove the BOS token -- this will get added when tokenizing, if necessary
        if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
            template = template.replace(tokenizer.bos_token, "")
        before_str = template.split("{optim_str_before}")[0]
        after_str = template.split("{optim_str_after} \n\n")[-1]

        target = " " + target if config.add_space_before_target else target
        target = target + tokenizer.eos_token

        # Tokenize everything that doesn't get optimized
        examples_tokenized   = _tokenize_fn([before_str], tokenizer, frontend_delimiters=self.frontend_delimiter, compute_gate=True)
        before_ids           = examples_tokenized["input_ids"][0].unsqueeze(0).to(model.device, torch.int64)
        before_expert_labels = examples_tokenized["expert_labels"][0].unsqueeze(0).to(model.device, torch.int64)

        examples_tokenized   = _tokenize_fn([enclosed_instruction], tokenizer, frontend_delimiters=self.frontend_delimiter, compute_gate=True, add_special_tokens=False)
        middle_ids           = examples_tokenized["input_ids"][0].unsqueeze(0).to(model.device, torch.int64)
        middle_expert_labels = torch.ones_like(middle_ids).to(model.device, torch.int64)

        examples_tokenized  = _tokenize_fn([after_str], tokenizer, frontend_delimiters=self.frontend_delimiter, compute_gate=True, add_special_tokens=False)
        after_ids           = examples_tokenized["input_ids"][0].unsqueeze(0).to(model.device, torch.int64)
        after_expert_labels = examples_tokenized["expert_labels"][0].unsqueeze(0).to(model.device, torch.int64)

        examples_tokenized = _tokenize_fn([target], tokenizer, frontend_delimiters=self.frontend_delimiter, compute_gate=False, add_special_tokens=False)
        target_ids           = examples_tokenized["input_ids"][0].unsqueeze(0).to(model.device, torch.int64)
        target_expert_labels = 2 * torch.ones_like(target_ids).to(model.device, torch.int64)

        # Embed everything that doesn't get optimized
        embedding_layer = self.embedding_layer
        before_embeds, after_embeds, target_embeds, middle_embeds \
            = [embedding_layer(ids) for ids in (before_ids, after_ids, target_ids, middle_ids)]

        self.target_ids = target_ids
        self.before_embeds = before_embeds
        self.after_embeds = after_embeds
        self.target_embeds = target_embeds
        self.before_expert_labels = before_expert_labels
        self.after_expert_labels = after_expert_labels
        self.target_expert_labels = target_expert_labels
        self.middle_embeds = middle_embeds
        self.middle_expert_labels = middle_expert_labels

        buffer    = self.init_buffer()
        optim_ids_before, optim_ids_after = buffer.get_best_ids()
        n_optim_tokens_before, n_optim_tokens_after = optim_ids_before.size(1), optim_ids_after.size(1)

        losses: List[float] = []
        optim_strings: List[Tuple[str, str]] = []

        for step in tqdm(range(config.num_steps)):

            # Compute the token gradient
            optim_ids_onehot_grad = self.compute_token_gradient(optim_ids_before, optim_ids_after)

            with torch.no_grad():
                # Sample candidate token sequences based on the token gradient
                sampled_ids = sample_ids_from_grad(
                    torch.cat([optim_ids_before.squeeze(0), optim_ids_after.squeeze(0)]),
                    optim_ids_onehot_grad.squeeze(0),
                    config.search_width,
                    config.topk,
                    config.n_replace,
                    not_allowed_ids=self.not_allowed_ids,
                ) # (search_width, n_before + n_after)

                if config.filter_ids: # filter unallowed ids
                    sampled_ids = filter_ids(sampled_ids, tokenizer) # (new_search_width, n_optim_ids)
                new_search_width = sampled_ids.shape[0]

                # Compute loss on all candidate sequences
                batch_size = new_search_width if config.batch_size is None else config.batch_size
                sample_ids_embed = embedding_layer(sampled_ids)
                sample_ids_expert_labels = torch.ones_like(sampled_ids).to(sampled_ids.device)

                input_embeds = torch.cat([
                    self.before_embeds.repeat(new_search_width, 1, 1),
                    sample_ids_embed[:, :n_optim_tokens_before],
                    self.middle_embeds.repeat(new_search_width, 1, 1),
                    sample_ids_embed[:, n_optim_tokens_before:],
                    self.after_embeds.repeat(new_search_width, 1, 1),
                    self.target_embeds.repeat(new_search_width, 1, 1),
                ], dim=1)

                if self.pass_expert_labels:
                    input_expert_labels = torch.cat(
                        [
                            self.before_expert_labels.repeat(new_search_width, 1), # (new_search_width, L_before_str)
                            sample_ids_expert_labels[:, :n_optim_tokens_before], # (new_search_width, n_optim_ids)
                            self.middle_expert_labels.repeat(new_search_width, 1),
                            sample_ids_expert_labels[:, n_optim_tokens_before:],
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
                optim_ids_before, optim_ids_after = optim_ids[:, :n_optim_tokens_before],  optim_ids[:, n_optim_tokens_before:]

                # Update the buffer based on the loss
                losses.append(current_loss)

            if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                buffer.add(current_loss, optim_ids_before, optim_ids_after)

            optim_ids_before, optim_ids_after = buffer.get_best_ids()
            optim_str_before = tokenizer.batch_decode(optim_ids_before)[0]
            optim_str_after = tokenizer.batch_decode(optim_ids_after)[0]
            optim_strings.append((optim_str_before, optim_str_after))
            # if self.config.early_stop:
            #     full_optim_str = template.replace("{optim_str_before}", optim_str_before).replace("{optim_str_after}", optim_str_after)
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
            #     logger.info(f"optim_str_before: {optim_str_before}, optim_str_after: {optim_str_after}, generated_str: {generated_str}")
            if self.stop_flag:
                logger.info("Early stopping due to finding a perfect match.")
                break

        min_loss_index = losses.index(min(losses))

        result = NeuralExecResult(
            best_loss=losses[min_loss_index],
            best_string=optim_strings[min_loss_index],
            losses=losses,
            strings=optim_strings,
        )

        return result

    def init_buffer(self) -> NeuralExecAttackBuffer:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        logger.info(f"Initializing attack buffer of size {config.buffer_size}...")

        # Create the attack buffer and initialize the buffer ids
        buffer = NeuralExecAttackBuffer(config.buffer_size)

        init_optim_ids_before = tokenizer(config.optim_str_init_before,
                                          add_special_tokens=False,
                                          return_tensors="pt")["input_ids"].to(model.device)
        if config.buffer_size > 1:
            init_buffer_ids_before = tokenizer(INIT_CHARS, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze().to(model.device)
            init_indices = torch.randint(0, init_buffer_ids_before.shape[0], (config.buffer_size - 1, init_optim_ids_before.shape[1]))
            init_buffer_ids_before = torch.cat([init_optim_ids_before, init_buffer_ids_before[init_indices]], dim=0)
        else:
            init_buffer_ids_before = init_optim_ids_before

        init_optim_ids_after = tokenizer(config.optim_str_init_after,
                                         add_special_tokens=False,
                                         return_tensors="pt")["input_ids"].to(model.device)
        if config.buffer_size > 1:
            init_buffer_ids_after = tokenizer(INIT_CHARS, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze().to(model.device)
            init_indices = torch.randint(0, init_buffer_ids_after.shape[0], (config.buffer_size - 1, init_optim_ids_after.shape[1]))
            init_buffer_ids_after = torch.cat([init_optim_ids_after, init_buffer_ids_after[init_indices]], dim=0)
        else:
            init_buffer_ids_after = init_optim_ids_after

        true_buffer_size = max(1, config.buffer_size)

        # Compute the loss on the initial buffer entries
        init_buffer_embeds = torch.cat([
            self.before_embeds.repeat(true_buffer_size, 1, 1),
            self.embedding_layer(init_buffer_ids_before),
            self.middle_embeds.repeat(true_buffer_size, 1, 1),
            self.embedding_layer(init_buffer_ids_after),
            self.after_embeds.repeat(true_buffer_size, 1, 1),
            self.target_embeds.repeat(true_buffer_size, 1, 1),
        ], dim=1)

        if self.pass_expert_labels:
            input_expert_labels = torch.cat(
                [
                    self.before_expert_labels.repeat(true_buffer_size, 1),
                    torch.ones_like(init_buffer_ids_before).to(init_buffer_ids_before.device),
                    self.middle_expert_labels.repeat(true_buffer_size, 1),
                    torch.ones_like(init_buffer_ids_after).to(init_buffer_ids_after.device),
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
            buffer.add(init_buffer_losses[i], init_buffer_ids_before[[i]], init_buffer_ids_after[[i]])

        buffer.log_buffer(tokenizer)
        logger.info("Initialized attack buffer.")

        return buffer

    def compute_token_gradient(
        self,
        optim_ids_before: Tensor,
        optim_ids_after: Tensor,
    ) -> Tensor:
        """Computes the gradient of the GCG loss w.r.t the one-hot token matrix.

        Args:
            optim_ids : Tensor, shape = (1, n_optim_ids)
                the sequence of token ids that are being optimized
        """
        model = self.model
        embedding_layer = self.embedding_layer

        # Create the one-hot encoding matrix of our optimized token ids
        all_optim_ids = torch.cat([optim_ids_before, optim_ids_after], dim=1) # (1, n_before+n_after)
        optim_ids_onehot = torch.nn.functional.one_hot(all_optim_ids, num_classes=embedding_layer.num_embeddings) # (1, n_before+n_after, vocab_size)
        optim_ids_onehot = optim_ids_onehot.to(model.device, model.dtype)
        optim_ids_onehot.requires_grad_()

        # (1, n_before+n_after, vocab_size) @ (vocab_size, embed_dim) -> (1, n_before+n_after, embed_dim)
        optim_embeds = optim_ids_onehot @ embedding_layer.weight

        input_embeds = torch.cat(
            [
                self.before_embeds,
                optim_embeds[:, :optim_ids_before.size(1)],
                self.middle_embeds,
                optim_embeds[:, optim_ids_before.size(1):],
                self.after_embeds,
                self.target_embeds,
            ],
            dim=1,
        )
        if self.pass_expert_labels:
            optim_ids_expert_labels = torch.ones_like(all_optim_ids).to(optim_embeds.device)
            input_expert_labels = torch.cat(
                [
                    self.before_expert_labels,
                    optim_ids_expert_labels[:, :optim_ids_before.size(1)],
                    self.middle_expert_labels,
                    optim_ids_expert_labels[:, optim_ids_before.size(1):],
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

        grads = torch.autograd.grad(outputs=[loss], inputs=[optim_ids_onehot])[0]

        return grads


# A wrapper around the GCG `run` method that provides a simple API
def run(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    messages: Union[str, List[dict]],
    enclosed_instruction: str,
    target: str,
    frontend_delimiter: str,
    config: Optional[NeuralExecConfig] = None,
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
        config = NeuralExecConfig()

    logger.setLevel(getattr(logging, config.verbosity))

    gcg = NeuralExec(model, tokenizer, config, pass_expert_labels,
              frontend_delimiter
    )
    result = gcg.run(messages, enclosed_instruction, target)
    return result