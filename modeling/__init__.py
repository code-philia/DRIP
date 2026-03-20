from .llama_instsep import LlamaForCausalLMMoE, LlamaForCausalLMMoEV2, LlamaMoEConfig
# from .llama_instfuse import LlamaForCausalLMFuse, LlamaForCausalLMConcatFuse, LlamaFuseConfig,\
#     LlamaForCausalLMConcatFuse, LlamaForCausalLMEmbeddingShift, LlamaForCausalLMNoFuse
from .llama_drip import LlamaForCausalLMFuse, LlamaFuseConfig, set_delimiter_ids_in_config
from .mistral_instsep import MistralForCausalLMMoE, MistralForCausalLMMoEV2, MistralMoEConfig
# from .mistral_instfuse import MistralForCausalLMFuse, MistralForCausalLMFuseV2, MistralFuseConfig
from .mistral_drip import MistralForCausalLMFuse, MistralFuseConfig
from .qwen_instsep import Qwen3MoEConfig, Qwen3ForCausalLMMoE, Qwen3ForCausalLMMoEV2
# from .qwen_instfuse import Qwen3FuseConfig, Qwen3ForCausalLMFuse
from .qwen_drip import Qwen3FuseConfig, Qwen3ForCausalLMFuse