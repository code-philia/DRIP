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
from data_generation.struq import _tokenize_fn, find_first_occurrence, find_last_occurrence
from .gcg import AttackBuffer, GCG, filter_ids, sample_ids_from_grad, GCGConfig, GCGResult
import torch.nn.functional as F

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


class RGCG(GCG):
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        config: GCGConfig,
        pass_expert_labels: bool,
        frontend_delimiter: str
    ):
        super().__init__(model, tokenizer, config, pass_expert_labels, frontend_delimiter)

        self._cached_target_hidden = None

    def _compute_target_hidden(self):
        """只在第一次调用时运行一次，缓存目标隐藏态。"""
        if self._cached_target_hidden is not None:
            return self._cached_target_hidden

        # 拼接 intended_instruction_embeds + target_embeds
        target_embeds = torch.cat([
            self.intended_instruction_embeds,
            self.target_embeds
        ], dim=1)
        if self.pass_expert_labels:
            target_expert_labels = torch.cat([
                self.intended_instruction_expert_labels,
                self.target_expert_labels
            ], dim=1)
            with torch.no_grad():
                out = self.model(
                    inputs_embeds=target_embeds,
                    expert_labels=target_expert_labels,
                    output_hidden_states=True,
                    return_dict=True
                )
        else:
            with torch.no_grad():
                out = self.model(
                    inputs_embeds=target_embeds,
                    output_hidden_states=True,
                    return_dict=True
                )

        pos = target_embeds.shape[1] - self.target_ids.shape[1] - 1 # (1, num_target_id, hidden_size)
        hidden = out.hidden_states[-1][:, pos, :].detach() # (1, hidden_size)
        self._cached_target_hidden = hidden
        del out
        gc.collect()
        torch.cuda.empty_cache()
        return hidden

    def run(
        self,
        message: str,
        intended_instruction: str,
        target: str,
    ) -> GCGResult:

        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        if config.seed is not None:
            set_seed(config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

        response_delimiter = DELIMITERS[self.frontend_delimiter][2]
        template = message.split(intended_instruction)[0].strip() + " {optim_str} " + intended_instruction + "\n\n" + response_delimiter + "\n"
        if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
            template = template.replace(tokenizer.bos_token, "")
        before_str, after_str = template.split(" {optim_str} ")

        target = target + tokenizer.eos_token

        #### The intended_instruction we want to inject
        data_delimiter = DELIMITERS[self.frontend_delimiter][1]
        intended_instruction_template      = message.split(data_delimiter)[0].strip() + "\n\n" + data_delimiter + "\n" + intended_instruction + "\n\n" + response_delimiter + "\n"
        examples_tokenized                 = _tokenize_fn([intended_instruction_template], tokenizer, frontend_delimiters=self.frontend_delimiter, compute_gate=True, add_special_tokens=False)
        intended_instruction_ids           = examples_tokenized["input_ids"][0].unsqueeze(0).to(model.device, torch.int64)
        intended_instruction_expert_labels = examples_tokenized["expert_labels"][0].unsqueeze(0).to(model.device, torch.int64)
        ################################################

        # Tokenize everything that doesn't get optimized
        examples_tokenized   = _tokenize_fn([before_str], tokenizer, frontend_delimiters=self.frontend_delimiter, compute_gate=True)
        before_ids           = examples_tokenized["input_ids"][0].unsqueeze(0).to(model.device, torch.int64)
        before_expert_labels = examples_tokenized["expert_labels"][0].unsqueeze(0).to(model.device, torch.int64)

        examples_tokenized  = _tokenize_fn([after_str], tokenizer, frontend_delimiters=self.frontend_delimiter, compute_gate=True, add_special_tokens=False)
        after_ids           = examples_tokenized["input_ids"][0].unsqueeze(0).to(model.device, torch.int64)
        after_expert_labels = examples_tokenized["expert_labels"][0].unsqueeze(0).to(model.device, torch.int64)
        after_expert_labels = after_expert_labels.masked_fill(after_expert_labels == 0, 1)

        examples_tokenized   = _tokenize_fn([target], tokenizer, frontend_delimiters=self.frontend_delimiter, compute_gate=False, add_special_tokens=False)
        target_ids           = examples_tokenized["input_ids"][0].unsqueeze(0).to(model.device, torch.int64)
        target_expert_labels = 2 * torch.ones_like(target_ids).to(model.device, torch.int64)

        # Embed everything that doesn't get optimized
        embedding_layer = self.embedding_layer
        before_embeds, after_embeds, target_embeds, intended_instruction_embeds = \
            [embedding_layer(ids) for ids in (before_ids, after_ids, target_ids, intended_instruction_ids)]

        self.target_ids = target_ids
        self.before_embeds = before_embeds
        self.after_embeds = after_embeds
        self.target_embeds = target_embeds
        self.intended_instruction_embeds = intended_instruction_embeds

        self.before_expert_labels = before_expert_labels
        self.after_expert_labels = after_expert_labels
        self.target_expert_labels = target_expert_labels
        self.intended_instruction_expert_labels = intended_instruction_expert_labels

        # Compute the hidden state Cache
        self._compute_target_hidden()

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
                batch_size = new_search_width if config.batch_size is None else config.batch_size

                input_embeds_to_search = torch.cat([
                    self.before_embeds.repeat(new_search_width, 1, 1),
                    embedding_layer(sampled_ids),
                    self.after_embeds.repeat(new_search_width, 1, 1),
                    self.target_embeds.repeat(new_search_width, 1, 1),
                ], dim=1)

                target_embeds_to_search = torch.cat(
                    [
                        self.intended_instruction_embeds.repeat(new_search_width, 1, 1),
                        self.target_embeds.repeat(new_search_width, 1, 1),
                    ],
                    dim=1,
                )

                if self.pass_expert_labels:
                    input_expert_labels_to_search = torch.cat(
                        [
                            self.before_expert_labels.repeat(new_search_width, 1),  # (new_search_width, L_before_str)
                            torch.ones_like(sampled_ids).to(sampled_ids.device),  # (new_search_width, n_optim_ids)
                            self.after_expert_labels.repeat(new_search_width, 1),
                            self.target_expert_labels.repeat(new_search_width, 1),
                        ],
                        dim=1,
                    )
                    target_expert_labels_to_search = torch.cat(
                        [
                            self.intended_instruction_expert_labels.repeat(new_search_width, 1),
                            self.target_expert_labels.repeat(new_search_width, 1),
                        ],
                        dim=1,
                    )
                    loss = find_executable_batch_size(self._compute_candidates_loss_original, batch_size)(input_embeds_to_search, target_embeds_to_search,
                                                                                                          input_expert_labels_to_search, target_expert_labels_to_search)
                else:
                    loss = find_executable_batch_size(self._compute_candidates_loss_original, batch_size)(input_embeds_to_search, target_embeds_to_search)

                current_loss = loss.min().item()
                optim_ids = sampled_ids[loss.argmin()].unsqueeze(0)

                # Update the buffer based on the loss
                losses.append(current_loss)
                if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                    buffer.add(current_loss, optim_ids)

            optim_ids = buffer.get_best_ids()
            optim_str = tokenizer.batch_decode(optim_ids)[0]
            optim_strings.append(optim_str)
            if self.config.early_stop:
                full_optim_str = template.replace("{optim_str}", optim_str)
                full_tokenized = _tokenize_fn([full_optim_str], tokenizer,
                                              frontend_delimiters=self.frontend_delimiter, compute_gate=True,
                                              add_special_tokens=False)
                full_ids = full_tokenized["input_ids"][0].unsqueeze(0).to(model.device, torch.int64)
                full_expert_labels = full_tokenized["expert_labels"][0].unsqueeze(0).to(model.device, torch.int64)
                generated_str = self.inference_check(
                    input_ids=full_ids,
                    input_expert_labels=full_expert_labels
                )
                if generated_str.lower().startswith(target.lower()):
                    self.stop_flag = True
                logger.info(f"optim_str: {optim_str}, generated_str: {generated_str}, optim_loss: {buffer.get_lowest_loss()}")

            buffer.log_buffer(tokenizer)

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
        init_buffer_embeds = torch.cat(
            [
            self.before_embeds.repeat(true_buffer_size, 1, 1),
            self.embedding_layer(init_buffer_ids),
            self.after_embeds.repeat(true_buffer_size, 1, 1),
            self.target_embeds.repeat(true_buffer_size, 1, 1),
            ],
            dim=1
        )

        target_embeds = torch.cat(
            [
                self.intended_instruction_embeds.repeat(true_buffer_size, 1, 1),
                self.target_embeds.repeat(true_buffer_size, 1, 1),
            ],
            dim=1,
        )

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
            target_expert_labels = torch.cat(
                [
                    self.intended_instruction_expert_labels.repeat(true_buffer_size, 1),
                    self.target_expert_labels.repeat(true_buffer_size, 1),
                ],
                dim=1,
            )
            init_buffer_losses = find_executable_batch_size(self._compute_candidates_loss_original, true_buffer_size)(init_buffer_embeds, target_embeds,
                                                                                                                      input_expert_labels, target_expert_labels)
        else:
            init_buffer_losses = find_executable_batch_size(self._compute_candidates_loss_original, true_buffer_size)(init_buffer_embeds, target_embeds)

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

        input_embeds = torch.cat(
            [
                self.before_embeds,
                optim_embeds,
                self.after_embeds,
                self.target_embeds,
            ],
            dim=1,
        )
        # 混合精度和一次大批量前向
        with torch.cuda.amp.autocast():
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
                current_output = model(
                    inputs_embeds=input_embeds,
                    expert_labels=input_expert_labels,
                    output_hidden_states=True,
                    return_dict=True
                )
            else:
                current_output = model(
                    inputs_embeds=input_embeds,
                    output_hidden_states=True,
                    return_dict=True
                )

        current_pos    = input_embeds.shape[1] - self.target_ids.shape[1] - 1 # (1, sequence_length, hidden_size)
        current_hidden = current_output.hidden_states[-1][:, current_pos, :]  # (1, hidden_size)

        # 获取目标隐藏态（已缓存）
        target_hidden = self._cached_target_hidden  # (1, num_target_id, hidden_size)

        loss = F.mse_loss(current_hidden, target_hidden, reduction="none").sum(-1)  # (1, )
        loss = loss.mean()

        optim_ids_onehot_grad = torch.autograd.grad(outputs=[loss], inputs=[optim_ids_onehot])[0]

        return optim_ids_onehot_grad

    def _compute_candidates_loss_original(
        self,
        search_batch_size: int,
        input_embeds: Tensor,
        target_embeds: Tensor,
        input_expert_labels: Optional[Tensor] = None,
        target_expert_labels: Optional[Tensor] = None
    ) -> Tensor:

        all_loss = []

        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i:i + search_batch_size]
                current_batch_size = input_embeds_batch.shape[0]

                if self.pass_expert_labels:
                    expert_labels_batch        = input_expert_labels[i:i + search_batch_size]
                    current_output = self.model(inputs_embeds=input_embeds_batch,
                                                expert_labels=expert_labels_batch,
                                                output_hidden_states=True, return_dict=True)
                else:
                    current_output = self.model(inputs_embeds=input_embeds_batch,
                                                output_hidden_states=True, return_dict=True)

                current_pos    = input_embeds.shape[1] - self.target_ids.shape[1] - 1
                current_hidden = current_output.hidden_states[-1][:, current_pos, :]  # (B, hidden_size)

                target_hidden = self._cached_target_hidden.expand(current_batch_size, -1)  # (B, hidden_size)

                loss = F.mse_loss(current_hidden, target_hidden, reduction="none").sum(-1)  # (B, )
                all_loss.append(loss)
                del current_output
                gc.collect()
                torch.cuda.empty_cache()

        return torch.cat(all_loss, dim=0)

# A wrapper around the GCG `run` method that provides a simple API
def run(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    message: str,
    intended_instruction: str,
    target: str,
    frontend_delimiter: str,
    config: Optional[GCGConfig] = None,
    pass_expert_labels: bool = False,
) -> GCGResult:
    """Generates a single optimized string using GCG.

    Args:
        model: The model to use for optimization.
        tokenizer: The model's tokenizer.
        target: The target generation.
        config: The GCG configuration to use.

    Returns:
        A GCGResult object that contains losses and the optimized strings.
    """
    if config is None:
        config = GCGConfig()

    logger.setLevel(getattr(logging, config.verbosity))

    gcg = RGCG(model, tokenizer, config, pass_expert_labels,
              frontend_delimiter
    )
    result = gcg.run(message, intended_instruction, target)
    return result