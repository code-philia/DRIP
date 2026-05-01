import os
os.environ["HF_HOME"] = "/mnt/sda/hf_hub"

import sys
import argparse
from typing import Optional, List, Dict, Any
import torch

# =============================================================================
# PATCH MARKER v2: bind device + pre-init PG with device_id BEFORE importing
# transformers. Just calling torch.cuda.set_device() is NOT enough — the
# PyTorch warning is triggered by init_process_group() being called without
# device_id, which HF Trainer does. We pre-initialize the PG ourselves so
# HF's later call is a no-op.
# =============================================================================
def _bind_and_init_distributed():
    if "LOCAL_RANK" not in os.environ:
        return  # single-process / no torchrun

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", local_rank))

    if not torch.cuda.is_available():
        return

    n_gpus = torch.cuda.device_count()
    assert local_rank < n_gpus, (
        f"LOCAL_RANK={local_rank} but only {n_gpus} GPUs visible. "
        f"Check CUDA_VISIBLE_DEVICES / nproc_per_node."
    )

    # 1. Bind CUDA current device for this process.
    torch.cuda.set_device(local_rank)

    # 2. Pre-init the default PG with device_id. This is the ONLY way to
    #    suppress the warning; HF Trainer's later init_process_group becomes
    #    a no-op since is_initialized() returns True.
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
        )

    # Print on every rank for diagnostic — if you don't see this in the log,
    # the patch is NOT running and you're executing a stale file.
    print(f"[PATCH v2][rank {rank}/{world_size}] bound cuda:{local_rank} "
          f"({torch.cuda.get_device_name(local_rank)}); "
          f"PG initialized={torch.distributed.is_initialized()}",
          flush=True)

_bind_and_init_distributed()
# =============================================================================

import transformers
from transformers import BitsAndBytesConfig

from config import DELIMITERS
from transformers.utils import logging
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import DPOTrainer
import torch.nn as nn
from training.trainer import DPOTrainerOurs, SFTTrainerOurs, SFTTrainerISE, DPOTrainerAIR, DPOArgsDRIP, \
    ModelArguments, DataArguments, TrainingArguments, AttackArguments, DPOArgsSecAlign
from data_generation.dpo_data_loader import make_dpo_data_module
from data_generation.data_loader import make_supervised_data_module, \
    make_supervised_data_module_orig, \
    smart_tokenizer_and_embedding_resize

from modeling import (
    MistralFuseConfig,
    MistralForCausalLMFuse,
    MistralISEConfig,
    MistralAIRConfig,
    MistralForCausalLMISE,
    MistralForCausalLMPFT,
    MistralForCausalLMAIR,
    LlamaISEConfig,
    LlamaAIRConfig,
    LlamaForCausalLMISE,
    LlamaForCausalLMPFT,
    LlamaForCausalLMAIR,
    Qwen3FuseConfig,
    Qwen3ForCausalLMFuse,
    LlamaFuseConfig,
    LlamaForCausalLMFuse,
    set_delimiter_ids_in_config,
    LlamaForCausalLMNoFuse,
    LlamaForCausalLMConcatFuse,
    LlamaForCausalLMEmbeddingShift
)

from config import DEFAULT_TOKENS, SPECIAL_DELM_TOKENS

logger = logging.get_logger(__name__)
os.environ.setdefault("WANDB_WATCH", "false")  # don't trace frozen base params



SHIFTS_ONLY_ARCHES = {"ise", "air"}

def pick_llama_model(arch: str):
    if arch == "fuse":
        return LlamaFuseConfig, LlamaForCausalLMFuse
    if arch == "nofuse":
        return LlamaFuseConfig, LlamaForCausalLMNoFuse
    if arch == "concatfuse":
        return LlamaFuseConfig, LlamaForCausalLMConcatFuse
    if arch == "embeddingshift":
        return LlamaFuseConfig, LlamaForCausalLMEmbeddingShift
    if arch == "ise":
        return LlamaISEConfig, LlamaForCausalLMISE
    if arch == "air":
        return LlamaAIRConfig, LlamaForCausalLMAIR
    if arch == "possep":
        return LlamaISEConfig, LlamaForCausalLMPFT
    raise ValueError(f"Unsupported LLaMA arch: {arch}")


def pick_mistral_model(arch: str):
    if arch == "fuse":
        return MistralFuseConfig, MistralForCausalLMFuse
    if arch == "ise":
        return MistralISEConfig, MistralForCausalLMISE
    if arch == "air":
        return MistralAIRConfig, MistralForCausalLMAIR
    if arch == "possep":
        return MistralISEConfig, MistralForCausalLMPFT
    raise ValueError(f"Unsupported Mistral arch: {arch}")


def pick_qwen_model(arch: str):
    if arch == "fuse":
        return Qwen3FuseConfig, Qwen3ForCausalLMFuse
    raise ValueError(f"Unsupported Qwen3 arch: {arch}")


def build_lora_config(objective: str, arch: str) -> LoraConfig:
    """Used for non-shifts-only arches (fuse / nofuse / concatfuse / etc.)."""
    if objective in ("dpo", "sft") and arch in ("fuse", "nofuse", "concatfuse", "embeddingshift"):
        modules = ["embed_tokens", "lm_head", "deinstruction_shift"]
        target_module = "all-linear"
    else:
        modules = ["lm_head", "embed_tokens"]
        target_module = "all-linear"

    return LoraConfig(
        r=16, lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_module,
        modules_to_save=modules,
    )


# Substrings used to identify trainable parameters in the shifts-only path.
# A parameter is trainable iff its qualified name contains any of these.
_TRAINABLE_KEYS = ("input_shifts", "intermediate_shifts", "lm_head")


def _is_trainable_name(name: str) -> bool:
    return any(k in name for k in _TRAINABLE_KEYS)


def freeze_base_train_shifts(model: nn.Module) -> int:
    """Freeze all params; unfreeze input_shifts / intermediate_shifts /
    lm_head / embed_tokens. Returns number of trainable parameters."""
    for p in model.parameters():
        p.requires_grad = False

    n_unfrozen = 0
    matched_names = []
    for name, p in model.named_parameters():
        if _is_trainable_name(name):
            p.requires_grad = True
            n_unfrozen += p.numel()
            matched_names.append(name)

    # Tied embedding sanity check
    tie = bool(getattr(model.config, "tie_word_embeddings", False))

    print("=== Shifts-only training: trainable params ===")
    for n in matched_names:
        p = dict(model.named_parameters())[n]
        print(f"  {n} | shape={tuple(p.shape)}")
    print(f"tie_word_embeddings: {tie}")
    print(f"Total trainable: {n_unfrozen:,} ({n_unfrozen / 1e6:.3f}M)")
    return n_unfrozen


def save_shifts_only(model: nn.Module, output_dir: str, tokenizer, is_main_process: bool) -> None:
    """Save only the trained tensors (shifts + lm_head + embed_tokens) + config + tokenizer.
    Avoids saving the frozen 7B base attention/mlp/norm weights."""
    if not is_main_process:
        return

    os.makedirs(output_dir, exist_ok=True)

    # Walk model and collect all trainable tensors (CPU copy).
    custom_state: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if _is_trainable_name(name):
            custom_state[name] = p.detach().cpu()

    save_path = os.path.join(output_dir, "shifts.bin")
    torch.save(custom_state, save_path)
    total_bytes = sum(t.numel() * t.element_size() for t in custom_state.values())
    print(f"Saved {len(custom_state)} trained tensors "
          f"({total_bytes / 1e6:.1f} MB) -> {save_path}")

    # Save config and tokenizer so inference can rebuild the model.
    base = getattr(model, "base_model", model)
    base_config = getattr(base, "config", None) or model.config
    base_config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved config and tokenizer -> {output_dir}")


def main(argv: Optional[List[str]] = None):
    # Stage 1: route by objective/arch
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--objective", choices=["dpo", "sft", "secalign_dpo", "struq_sft", "air_dpo"], required=True)
    parser.add_argument("--model-family", choices=["llama", "mistral", "qwen"], required=False, default="base")
    parser.add_argument("--arch", required=True, choices=["base", "fuse", "nofuse", "concatfuse", "embeddingshift", "ise", "possep", "air"])
    known, remaining = parser.parse_known_args(argv)

    # Stage 2: parse HF dataclasses per flow
    if known.objective == "dpo":
        hf_parser = transformers.HfArgumentParser((ModelArguments, DataArguments, DPOArgsDRIP, AttackArguments))
        model_args, data_args, training_args, attack_args = hf_parser.parse_args_into_dataclasses(remaining)
    elif known.objective == "air_dpo":
        hf_parser = transformers.HfArgumentParser((ModelArguments, DataArguments, DPOArgsSecAlign, AttackArguments))
        model_args, data_args, training_args, attack_args = hf_parser.parse_args_into_dataclasses(remaining)
    elif known.objective == "secalign_dpo":
        hf_parser = transformers.HfArgumentParser((ModelArguments, DataArguments, DPOArgsSecAlign, AttackArguments))
        model_args, training_args, data_args, attack_args = hf_parser.parse_args_into_dataclasses(remaining)
    else:  # SFT
        hf_parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, AttackArguments))
        model_args, data_args, training_args, attack_args = hf_parser.parse_args_into_dataclasses(remaining)

    data_args.attack = getattr(attack_args, "attack", None)
    logger.info(f"Objective={known.objective} | Family={getattr(known, 'model_family', 'base')} | Arch={known.arch}")
    print("\n\n" + training_args.output_dir + "\n\n")

    # ------------------
    # Create tokenizer
    # ------------------
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=getattr(training_args, "cache_dir", None),
        model_max_length=getattr(training_args, "model_max_length", 512),
        padding_side=getattr(model_args, "padding_side", "right"),
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    frontend_delimiters = (data_args.attack or "").split("_")[0] if getattr(data_args, "attack", None) else ""

    # ------------------
    # Create model
    # ------------------
    if known.arch == "base" or known.objective in ("secalign_dpo", "struq_sft"):
        config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=getattr(training_args, "cache_dir", None),
            config=config,
            low_cpu_mem_usage=True,        # init on meta device, save ~16GB CPU RAM/rank
            torch_dtype=torch.bfloat16,    # avoid fp32 intermediate copy
        )
        special_tokens_dict = dict()
        special_tokens_dict["pad_token"] = DEFAULT_TOKENS.get('pad_token', tokenizer.pad_token)
        special_tokens_dict["eos_token"] = DEFAULT_TOKENS.get('eos_token', tokenizer.eos_token)
        special_tokens_dict["bos_token"] = DEFAULT_TOKENS.get('bos_token', tokenizer.bos_token)
        special_tokens_dict["unk_token"] = DEFAULT_TOKENS.get('unk_token', tokenizer.unk_token)
        special_tokens_dict["additional_special_tokens"] = SPECIAL_DELM_TOKENS
        smart_tokenizer_and_embedding_resize(special_tokens_dict=special_tokens_dict,
                                             tokenizer=tokenizer,
                                             model=model)
    else:
        if known.model_family == "llama":
            Cfg, Model = pick_llama_model(known.arch)
        elif known.model_family == "mistral":
            Cfg, Model = pick_mistral_model(known.arch)
        elif known.model_family == "qwen":
            Cfg, Model = pick_qwen_model(known.arch)
        else:
            raise ValueError(f"Unsupported model-family: {known.model_family}")

        config = Cfg.from_pretrained(model_args.model_name_or_path)
        if known.arch == "air":
            config.apply_intermediate_shifts = True
        num_labels = len(DELIMITERS[frontend_delimiters])
        if num_labels == 4:
            set_delimiter_ids_in_config(config, tokenizer,
                                        inst_delm=DELIMITERS[frontend_delimiters][1],
                                        data_delm=DELIMITERS[frontend_delimiters][2],
                                        response_delm=DELIMITERS[frontend_delimiters][3],
                                        num_labels=num_labels)
        else:
            set_delimiter_ids_in_config(config, tokenizer,
                                        inst_delm=DELIMITERS[frontend_delimiters][0],
                                        data_delm=DELIMITERS[frontend_delimiters][1],
                                        response_delm=DELIMITERS[frontend_delimiters][-1],
                                        num_labels=num_labels)
        model = Model.from_pretrained(
            model_args.model_name_or_path,
            ignore_mismatched_sizes=True,
            config=config,
            low_cpu_mem_usage=True,        # init on meta device, save ~16GB CPU RAM/rank
            torch_dtype=torch.bfloat16,    # avoid fp32 intermediate copy
        )

    if known.arch in SHIFTS_ONLY_ARCHES:
        freeze_base_train_shifts(model)
    else:
        lora_config = build_lora_config(model, known.objective, known.arch)
        model = get_peft_model(model, lora_config)

    # ------------------
    # Data module
    # ------------------
    if known.objective == "dpo":
        data_module = make_dpo_data_module(tokenizer=tokenizer,
                                           data_args=data_args,
                                           frontend_delimiters=frontend_delimiters)
    elif known.objective in ["secalign_dpo", "air_dpo"]:
        train_dataset = load_dataset('json',
                                     data_files=data_args.data_path_list,
                                     split="train")
        data_module = {"train_dataset": train_dataset, "eval_dataset": None}
    elif known.objective == "sft":
        data_module = make_supervised_data_module(tokenizer=tokenizer,
                                                  data_args=data_args,
                                                  frontend_delimiters=frontend_delimiters,
                                                  downsample=getattr(training_args, "downsample", False))
        if not getattr(training_args, "downsample", True) and getattr(training_args, "lr_scale", True):
            training_args.learning_rate /= data_module["train_dataset"].data_copy_count
    else:  # struq_sft
        data_module = make_supervised_data_module_orig(tokenizer=tokenizer,
                                                       data_args=data_args,
                                                       frontend_delimiters=frontend_delimiters,
                                                       downsample=getattr(training_args, "downsample", False))
        if not getattr(training_args, "downsample", True) and getattr(training_args, "lr_scale", True):
            training_args.learning_rate /= data_module["train_dataset"].data_copy_count

    # Optional window attribute
    if hasattr(model_args, "window_size") and getattr(model_args, "window_size", 0) > 0 and hasattr(model.config, "window"):
        model.config.window = model_args.window_size


    # ------------------
    # Trainer selection
    # ------------------
    if known.objective == "dpo":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        ref_model = Model.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,        # avoid fp32 intermediate copy in CPU
            device_map={"": int(os.environ.get("LOCAL_RANK", "0"))},  # load straight to this rank's GPU
        )
        ref_model.config.use_cache = False  # no KV cache buffer for ref forward
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        trainer = DPOTrainerOurs(
            model=model,
            ref_model=ref_model,
            processing_class=tokenizer,
            args=training_args,
            **data_module
        )
    elif known.objective == "air_dpo":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        ref_model = Model.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,        # avoid fp32 intermediate copy in CPU
            device_map={"": int(os.environ.get("LOCAL_RANK", "0"))},  # load straight to this rank's GPU
        )
        ref_model.config.use_cache = False  # no KV cache buffer for ref forward
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        trainer = DPOTrainerAIR(
            model=model,
            ref_model=ref_model,
            args=training_args,
            processing_class=tokenizer,
            **data_module
        )
    elif known.objective == "secalign_dpo":
        trainer = DPOTrainer(
            model,
            args=training_args,
            processing_class=tokenizer,
            **data_module,
        )
    else:  # SFT objective
        if known.arch in ("ise",):
            trainer_class = SFTTrainerISE
        elif known.arch in ("fuse", "nofuse", "concatfuse", "embeddingshift"):
            trainer_class = SFTTrainerOurs
        else:
            trainer_class = transformers.Trainer
        trainer = trainer_class(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            **data_module
        )

    # Debug: print trainable params summary (PEFT path only; shifts-only
    # already prints via freeze_base_train_shifts).
    if hasattr(trainer.model, "print_trainable_parameters"):
        print("=== Trainable params (PEFT) ===")
        trainer.model.print_trainable_parameters()

    # ------------------
    # Train
    # ------------------
    # Resume support: honor --resume_from_checkpoint from CLI.
    #   - If a path is given, resume from that exact checkpoint dir.
    #   - If "True"/True is given, auto-detect the latest checkpoint in output_dir.
    #   - Otherwise (None / False / "False"), start fresh.
    resume_arg = getattr(training_args, "resume_from_checkpoint", None)

    def _truthy(v):
        return isinstance(v, str) and v.strip().lower() in ("true", "1", "yes")

    if resume_arg is None or resume_arg is False or (isinstance(resume_arg, str) and resume_arg.strip().lower() in ("false", "0", "no", "")):
        resume_value = None
    elif resume_arg is True or _truthy(resume_arg):
        # Auto-detect: let Trainer find latest checkpoint-* under output_dir.
        ckpts = []
        if os.path.isdir(training_args.output_dir):
            for d in os.listdir(training_args.output_dir):
                if d.startswith("checkpoint-") and os.path.isdir(os.path.join(training_args.output_dir, d)):
                    try:
                        ckpts.append((int(d.split("-")[1]), d))
                    except ValueError:
                        pass
        if ckpts:
            resume_value = True
            latest = max(ckpts)[1]
            if trainer.is_world_process_zero():
                logger.info(f"[resume] Auto-resuming from latest checkpoint: {latest}")
        else:
            resume_value = None
            if trainer.is_world_process_zero():
                logger.info(f"[resume] --resume_from_checkpoint=True but no checkpoint-* found in "
                            f"{training_args.output_dir}; starting fresh.")
    else:
        # Treat as explicit path
        resume_value = resume_arg
        if trainer.is_world_process_zero():
            logger.info(f"[resume] Resuming from explicit checkpoint path: {resume_value}")

    trainer.train(resume_from_checkpoint=resume_value)

    if trainer.is_world_process_zero():
        logger.info("Training done. Saving...")

    # ------------------
    # Save
    #   shifts-only: save only the shifts tensors (DDP, no FSDP gather)
    #   PEFT path: standard trainer.save_model() + state
    # ------------------
    if known.arch in SHIFTS_ONLY_ARCHES:
        save_shifts_only(
            model=trainer.model,
            output_dir=training_args.output_dir,
            tokenizer=tokenizer,
            is_main_process=trainer.is_world_process_zero(),
        )
    else:
        trainer.save_model(output_dir=training_args.output_dir)
        trainer.save_state()
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)
            base_config = getattr(trainer.model, "base_model", trainer.model).config
            base_config.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/sda/hf_cache")
    main()