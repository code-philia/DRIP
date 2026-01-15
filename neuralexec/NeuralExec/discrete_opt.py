import math
import torch
import numpy as np
import tqdm
import random
import math

from .ex_triggers import NeuralExec
from .adv_prompts import AdvPrompt, Prompt
from .utility import *
from .semantic_checker import SemanticChecker
from .data import _tokenize_fn_batch
from typing import List
import gc
import torch.nn.functional as F
from torch import nn

class WhiteBoxTokensOpt:
    def __init__(
        self,
        llm_obj,
        hparams,
    ):
        self.llm_obj = llm_obj
        self.llm = llm_obj.llama_model
        self.tokenizer = llm_obj.tokenizer
        self.hparams = hparams
        self.delimiter = hparams["delimiter"]
        self.model_class = hparams['model_class']
        self.pass_expert_labels = True if self.model_class is not None else False

        self.device = self.llm.device
        
        if self.llm:
            self.emb_matrix = self.llm.llama_model.get_input_embeddings().weight
        
            self.loss_fun = torch.nn.CrossEntropyLoss(reduction='none')

            self.generate_args = {'max_new_tokens': 300, 'do_sample':False, 'temperature':.8}
            
            tokens_to_exclude = get_tokens_to_skip(
                llm_obj,
                skip_non_natural=hparams['skip_non_natural'],
                skip_non_ascii=hparams['skip_non_ascii'],
            )
            self.tokens_to_exclude_mask = torch.ones(self.emb_matrix.size(0), dtype=bool)
            self.tokens_to_exclude_mask[tokens_to_exclude] = False

        self.prompt_kargs = {'adv_pos':None}    
        self.adv_token_id = llm_obj.adv_token_id
        print(f'adv_token_id: {self.adv_token_id}')
        
        if 'emb_model' in self.hparams:
            self.sm = SemanticChecker(hparams)
        else:
            self.sm = None
        
    def make_model_input(self, prompts, nes, with_target=True, keep_placeholder_tokens=False):
        is_collection = lambda t: not t is None and (not issubclass(type(t), AdvPrompt)) and type(t) is list
        
        is_collection_prompt  = is_collection(prompts)
        is_collection_adv_tok = is_collection(nes)
        
        if is_collection_prompt and is_collection_adv_tok:
            raise Exception("Multi prompt and multi adv_toks not supported")
            
        if is_collection_prompt:
            return self._make_model_input_multi_prompt_single_adv_tok(prompts, nes, with_target=with_target, keep_placeholder_tokens=keep_placeholder_tokens)
            
        if is_collection_adv_tok:
            return self._make_model_input_single_prompt_multi_adv_tok(prompts, nes, with_target=with_target, keep_placeholder_tokens=keep_placeholder_tokens)

        return self._make_model_input_multi_prompt_single_adv_tok([prompts], nes, with_target=with_target, keep_placeholder_tokens=keep_placeholder_tokens)
    
    #
    # def prepare_labels(self, labels: List[str], max_length: int):
    #     labels_tok = self.tokenizer(labels,
    #                                 add_special_tokens=False,
    #                                 return_tensors="pt",
    #                                 max_length=max_length,
    #                                 padding='max_length').to(self.device)
    #     labels_tok.input_ids = torch.roll(labels_tok.input_ids, -1)
    #     labels_tok.attention_mask = torch.roll(labels_tok.attention_mask, -1)
    #     labels_tok.attention_mask[:,-1] = 0
    #     return labels_tok
    def prepare_labels(self, targets_str, max_length):
        B = len(targets_str)
        labels = torch.full(
            (B, max_length),
            -100,
            dtype=torch.long,
            device=self.device,
        )

        for i, t in enumerate(targets_str):
            ids = self.tokenizer(t, add_special_tokens=False)["input_ids"]
            ids = torch.tensor(ids, device=self.device, dtype=torch.long)
            L = ids.size(0)

            start = max_length - L  #
            labels[i, start:start + L] = ids  #

        return labels
        
    def embed_tokens(self, prompts_tok):
        return self.llm.llama_model.embed_tokens(prompts_tok)

    def get_logits(self, prompt, attention_mask, expert_labels):
        input_type = prompt.dtype
        if input_type == torch.int64:
            # input tokens:
            if self.pass_expert_labels:
                return self.llm(input_ids=prompt,
                                expert_labels=expert_labels,
                                attention_mask=attention_mask).logits
            else:
                return self.llm(input_ids=prompt,
                                attention_mask=attention_mask).logits
        else:
            # input embedded tokens:
            if self.pass_expert_labels:
                return self.llm(inputs_embeds=prompt,
                                expert_labels=expert_labels,
                                attention_mask=attention_mask).logits
            else:
                return self.llm(inputs_embeds=prompt,
                                attention_mask=attention_mask).logits
        
    def get_gradient(self, ne, prompts):
        # parse input
        prompts_tok, expert_labels, labels_tok, adv_mask, attention_mask = self.make_model_input(prompts, ne, keep_placeholder_tokens=True)
        batch_size = prompts_tok.input_ids.size(0)

        prompt_emb = self.embed_tokens(prompts_tok.input_ids)

        # make onehot
        # Build embeddings as before
        adv_ohe = F.one_hot(
            ne.tokens,  # [T_adv]
            num_classes=self.emb_matrix.shape[0],  # vocab size
        ).float().to(self.emb_matrix.device).to(self.emb_matrix.dtype)

        # repeat over batch, like your original code
        adv_ohe_rep = adv_ohe.repeat(batch_size, 1, 1)  # [B, T_adv, V]
        # 2) Map one-hot -> embeddings
        adv_emb = adv_ohe_rep @ self.emb_matrix  # [B, T_adv, d]
        # this is the thing we want gradients for
        adv_emb.requires_grad_(True)  # [T_adv, V]

        # Insert into prompt embeddings
        # (clone to avoid in-place issues on a tensor that might be used elsewhere)
        prompt_emb = prompt_emb.clone()  # [B, L, d]
        prompt_emb[adv_mask] = adv_emb.reshape(-1, self.emb_matrix.shape[1])  # [B, L, d] # fixme: check adv_mask.nonzero()

        self.llm.zero_grad()
        with torch.enable_grad():
            logits = self.get_logits(prompt_emb, attention_mask, expert_labels) # fixme: check prompts_tok.attention_mask[adv_mask]
            # slice logits?

            losses = self._compute_loss(logits, labels_tok)
            loss = losses.mean()

            # directly get dL/d(one_hot_adv)  (adv part only)
            grad_adv_emb = torch.autograd.grad(
                outputs=[loss],
                inputs=[adv_emb],  # NOTE: base [T_adv, V], not the repeated version
                retain_graph=False,
                create_graph=False
            )[0]
            gc.collect()

        grad_adv_ohe = grad_adv_emb.detach() @ self.emb_matrix.T
        grad = grad_adv_ohe.sum(dim=0)  # [V] fixme
        norm = grad.norm()
        if norm > 0:
            grad = grad / norm

        return grad, loss, losses
        
    
    def _make_model_input_multi_prompt_single_adv_tok(self, prompts, ne, keep_placeholder_tokens, with_target=True):

        # finalize prompts
        prompts_str = [prompt(self.llm_obj, ne, with_target=with_target, **self.prompt_kargs) for prompt in prompts] # fixme
        prompts_tok, expert_labels, expert_labels_lens, attention_mask = _tokenize_fn_batch(
            prompts_str,
            self.tokenizer,
            frontend_delimiters=self.delimiter,
            compute_gate=True,
        )
        # move everything to device
        prompts_tok = prompts_tok.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if expert_labels is not None:
            expert_labels = expert_labels.to(self.device)

        adv_mask   = prompts_tok.input_ids == self.adv_token_id
        max_length = prompts_tok.input_ids.size(1)

        if not keep_placeholder_tokens and not ne is None:
            # replace adv tokens
            prompts_tok.input_ids[adv_mask] = ne.tokens.repeat((len(prompts), 1)).ravel()

        # just to compute the label shift
        targets_str = [prompt.target[self.llm_obj.llm_name] for prompt in prompts]
        labels_tok = self.prepare_labels(targets_str, max_length)

        return prompts_tok, expert_labels, labels_tok, adv_mask, attention_mask

        
    def _make_model_input_single_prompt_multi_adv_tok(self, prompt, nes, keep_placeholder_tokens, with_target):

        # finalize prompts
        prompts_str = [prompt(self.llm_obj, ne, with_target=with_target, **self.prompt_kargs) for ne in nes]
        prompts_tok, expert_labels, expert_labels_lens, attention_mask  = _tokenize_fn_batch(
            prompts_str,
            self.tokenizer,
            frontend_delimiters=self.delimiter,
            compute_gate=True,
        )
        # move everything to device
        prompts_tok = prompts_tok.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if expert_labels is not None:
            expert_labels = expert_labels.to(self.device)

        adv_mask = prompts_tok.input_ids == self.adv_token_id
        max_length = prompts_tok.input_ids.size(1)

        if not keep_placeholder_tokens:
            adv_toks = torch.concat([ne.tokens for ne in nes])
            prompts_tok.input_ids[adv_mask] = adv_toks
    
        # just to compute the label shift
        targets_str = [prompt.target[self.llm_obj.llm_name] for i in range(len(nes))]
        labels_tok = self.prepare_labels(targets_str, max_length)
    
        return prompts_tok, expert_labels, labels_tok, adv_mask, attention_mask

    @staticmethod
    def make_labels_weights(labels_tok):
        w = labels_tok.attention_mask.flip(1).cumsum(1).flip(1) * labels_tok.attention_mask
        w = w ** 2
        w = w / w.sum(1, keepdim=True)
        return w
        
    # def _compute_loss(self, logits, targets, weighted=True):
    #     # compute loss
    #     all_loss = self.loss_fun(torch.permute(logits, (0, 2, 1)), targets.input_ids)
    #
    #     if weighted:
    #         W = self.make_labels_weights(targets)
    #         label_loss = all_loss * W
    #         losses = label_loss.sum(1)
    #     else:
    #         # mask (+weight) only label loss
    #         label_loss = all_loss * targets.attention_mask
    #         # per example avg
    #         losses = (label_loss.sum(1) / targets.attention_mask.sum(1))
    #
    #     return losses

    def _compute_loss(self, logits, labels):
        # logits: [B, L, V], labels: [B, L]
        B, L, V = logits.shape

        shift_logits = logits[:, :-1, :].contiguous()  # [B, L-1, V]
        shift_labels = labels[:, 1:].contiguous()  # [B, L-1]

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        loss = loss_fct(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
        ).view(B, -1)  # [B, L-1]

        mask = (shift_labels != -100)  #
        denom = mask.sum(dim=1).clamp(min=1)
        loss_per_seq = (loss * mask).sum(dim=1) / denom

        return loss_per_seq
        
    @torch.no_grad()
    def sample_new_candidates(self, ne, grad):
        # from Universal and transferable adversarial attacks on aligned language models
        m    = self.hparams['m']
        topk = self.hparams['topk_probability_new_candidate']
        batch_size = self.hparams['new_candidate_pool_size']

        control_toks = ne.tokens
        grad[:, ~self.tokens_to_exclude_mask] = torch.inf

        top_indices = (-grad).topk(topk, dim=1).indices
        original_control_toks = control_toks.repeat(batch_size, 1)
        pre = original_control_toks.clone()

        for _ in range(m):

            new_token_pos = torch.randint(0, len(control_toks), (batch_size,), device=grad.device).type(torch.int64)

            new_token_val = torch.gather(
                top_indices[new_token_pos], 1, 
                torch.randint(0, topk, (batch_size, 1),
                device=grad.device)
            )

            new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)

        nes  = [NeuralExec(ne.prefix.to(self.device), ne.postfix.to(self.device), ne.sep)]
        nes += [NeuralExec(adv_tok[:ne.prefix_size], adv_tok[ne.prefix_size:], sep=ne.sep) for adv_tok in new_control_toks]

        return nes

    
    def random_toks(self, n):
        all_tokens = torch.arange(0, self.emb_matrix.size(0))
        all_tokens = all_tokens[self.tokens_to_exclude_mask]
        idx = torch.multinomial(torch.ones(all_tokens.size(0)), num_samples=n, replacement=True)

        tokens = all_tokens[idx].to(self.device)
        return tokens
    
    
    def init_adv_seg(self, prefix_size, postfix_size, sep=''):
        
        # It sucks, I know. But tokenizers suck too
        while True:
            prefix  = self.random_toks(prefix_size)
            postfix = self.random_toks(postfix_size)
            ne = NeuralExec(prefix, postfix, sep=sep)
            ne(self.tokenizer)
            
            if ne.prefix_size == prefix_size and ne.postfix_size == postfix_size:
                return ne

    def init_adv_seg_boot(self, prefix_str, postfix_str, sep):
        
        prefix  = self.tokenizer(prefix_str, return_tensors='pt', add_special_tokens=False).input_ids[0].to(self.device)
        postfix = self.tokenizer(postfix_str, return_tensors='pt', add_special_tokens=False).input_ids[0].to(self.device)
        
        ne = NeuralExec(prefix, postfix, sep=sep)
        ne(self.tokenizer)
        print(ne.prefix_size, ne.postfix_size)
            
        return ne

    def _eval_loss(self, prompt, nes, weighted=True):
        batch_size = self.hparams['batch_size_eval']
        losses = []

        num_batches = math.ceil(len(nes) / batch_size)

        for i in range(num_batches):
            start = batch_size * i
            stop = batch_size * (i+1)

            prompts_tok, expert_labels, labels_tok, _, attention_mask = self.make_model_input(prompt, nes=nes[start:stop])

            with torch.no_grad():
                logits = self.get_logits(prompts_tok.input_ids, attention_mask, expert_labels)
                loss = self._compute_loss(logits, labels_tok)

            loss = loss.detach().cpu().numpy()
            losses.append(loss)

        losses = np.concatenate(losses)

        return losses
    
    
    @torch.no_grad()
    def test_candidates(self, prompts, nes):
        
        n = self.hparams.get('#prompts_to_sample_for_eval', -1)
        if n > 0:
            random.shuffle(prompts)
            prompts = prompts[:n]
        
        losses = []
        for prompt in tqdm.tqdm(prompts):
            _losses = self._eval_loss(prompt, nes)
            losses.append(_losses)
            
        losses = np.concatenate([loss[np.newaxis,:] for loss in losses])
        agg_losses = losses.mean(0)
        
        if not self.sm is None:
            print("Comp. Semantic distraction...")
            emb_losses = self.sm(nes, prompts, self.tokenizer)
            print(agg_losses.mean(), emb_losses.mean())
            agg_losses += emb_losses
        
        best_i = agg_losses.argmin()        
        best_loss = agg_losses[best_i]
        best_candidate = nes[best_i]

        return best_candidate, best_loss, agg_losses, losses  


    def get_gradient_accum(self, ne, prompts):
        
        accum_grad, accum_loss, _losses = self.get_gradient(ne, [prompts[0]])
        losses = [_losses.detach().cpu().numpy()]
        for prompt in prompts[1:]:
            _grad, _loss, _losses = self.get_gradient(ne, [prompt])
            accum_grad += _grad
            losses.append(_losses.detach().cpu().numpy())
        
        accum_grad /= len(prompts)
        
        losses = np.concatenate(losses)
        accum_loss = losses.mean()       
            
        return accum_grad, accum_loss, losses
    
    @torch.no_grad()
    def eval_loss(self, prompts, ne):
        
        batch_size = self.hparams['batch_size_eval']
        num_batches = math.ceil(len(prompts) / batch_size)
        
        _losses = []
        
        for i in tqdm.trange(num_batches):
            start = batch_size * i
            stop = batch_size * (i+1)

            prompts_tok, expert_labels, labels_tok, _, attention_mask = self.make_model_input(prompts[start:stop], ne) # fixme
            prompts_tok = prompts_tok.to(self.device)
            input_ids = prompts_tok.input_ids

            if self.pass_expert_labels:
                logits = self.llm(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  expert_labels=expert_labels).logits
            else:
                logits = self.llm(input_ids=input_ids,
                                  attention_mask=attention_mask).logits
            losses = self._compute_loss(logits, labels_tok)

            losses = losses.detach().cpu().numpy()
            _losses.append(losses)

        _losses = np.concatenate(_losses)

        return _losses
    
        
    def filter_candidates(self, control, new_candidate_tok):
        
        def is_ok(nne):
            prefix_s, postfix_s = nne(self.tokenizer)
            adv_seg = prefix_s+postfix_s
            
            if nne.prefix_size != control.prefix_size or nne.postfix_size != control.postfix_size:
                return False
            
            for s in SPECIALS_NON_ATOMIC:
                if s in adv_seg:
                    print(f"Filtered: {adv_seg}")
                    return False
            return True

        good_nes = list(filter(is_ok, new_candidate_tok))
                
        return good_nes 
