"""Offline model architecture registry, researched 2026-10-17.

MODEL_DEFAULTS is the requested list of model names and default structures.
Sequence length here means the selected checkpoint's unmodified configured
context limit, NOT a recommended fine-tuning length. Explicit inputs win.

Parameter counts are derived from the published dense architecture (including
embeddings and the LM head, counting tied weights once); they are not rounded
marketing sizes. The accounting below makes that derivation reproducible.
Sources are publisher configs/cards and publisher/reference implementations.
No models, weights, tokenizers, or network connections are loaded at runtime.
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Final


VERIFIED_ON: Final = "2026-10-17"
META_CONFIG_SOURCE: Final = "https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py"
META_31_CARD: Final = "https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md"
META_32_CARD: Final = "https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md"
META_33_CARD: Final = "https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/MODEL_CARD.md"
GEMMA_CONFIG_SOURCE: Final = "https://github.com/google/gemma_pytorch/blob/main/gemma/config.py"
GEMMA_MODEL_SOURCE: Final = "https://github.com/google/gemma_pytorch/blob/main/gemma/model.py"

# Workflow choices are separate from pretrained model architecture. A model
# cannot tell us a researcher's dataset size, training method, or micro-batch.
ESTIMATOR_DEFAULTS: Final = {
    "batch_size": 1,
    "training_method": "LoRA",
    "precision": "BF16",
    "epochs": 1,
}

INFERENCE_DEFAULTS: Final = {
    "concurrent_sequences": 1,
    "precision": "BF16",
    "kv_cache_precision": "BF16",
}

# Inference weights and KV cache are accounted for separately. These values
# are independent of full-fine-tuning bytes/parameter and optimizer states.
INFERENCE_WEIGHT_BYTES: Final = {
    "pure_fp32": 4, "bf16": 2, "fp16": 2,
    "fp8": 1, "int8": 1, "int4": 0.5,
}
INFERENCE_PRECISION_ALIASES: Final = {
    "purefp32": "pure_fp32", "fp32": "pure_fp32", "float32": "pure_fp32",
    "bf16": "bf16", "bfloat16": "bf16", "purebf16": "bf16", "bf16mixed": "bf16",
    "fp16": "fp16", "float16": "fp16", "purefp16": "fp16", "fp16mixed": "fp16",
    "fp8": "fp8", "float8": "fp8", "fp8mixed": "fp8",
    "int8": "int8", "int4": "int4",
}
# INFERENCE_KV_BYTES: Final = {"bf16": 2, "fp8": 1}
# INFERENCE_KV_ALIASES: Final = {
#     "bf16": "bf16", "bfloat16": "bf16", "fp8": "fp8", "float8": "fp8",
# }

# These names do not uniquely identify a public pretrained checkpoint.
# Supply explicit architecture inputs, or add a verified profile below.
UNRESOLVED_MODEL_NAMES: Final = ("LabLM", "HelpDeskLM", "MiniServe")


def _profile(
    model_name, checkpoint_id, *, layers, hidden, context, intermediate,
    heads, kv_heads, vocab, tied=False, head_dim=None, qkv_bias=False,
    norms_per_layer=2, qk_norm=False, checkpoint_dtype="bfloat16",
    aliases=(), sources=(), note="",
):
    """Construct one constant record and count unique dense parameters.

    All entries have a gated 3-projection MLP and bias-free output/MLP layers.
    Phi fused QKV and gate/up tensors have the same parameter count as separate
    projections. Qwen2 has Q/K/V biases; Qwen3 has per-head Q/K RMSNorm;
    Gemma 2 has four residual-stream RMSNorms per layer and tied embeddings.
    Gemma/Qwen3 head_dim can differ from hidden_size / attention_heads.
    """
    d = head_dim if head_dim is not None else hidden // heads
    attention = 2 * hidden * heads * d + 2 * hidden * kv_heads * d
    bias = (heads + 2 * kv_heads) * d if qkv_bias else 0
    mlp = 3 * hidden * intermediate
    norms = norms_per_layer * hidden + (2 * d if qk_norm else 0)
    embedding_and_head = vocab * hidden * (1 if tied else 2)
    parameters = embedding_and_head + layers * (attention + bias + mlp + norms) + hidden
    config_url = f"https://huggingface.co/{checkpoint_id}/blob/main/config.json" if checkpoint_id else None
    return {
        "model_name": model_name,
        "checkpoint_id": checkpoint_id,
        "parameter_count": parameters,
        "parameter_count_basis": "derived from published dense architecture; tied weights counted once",
        "number_of_layers": layers,
        "hidden_size": hidden,
        "sequence_length": context,
        "max_position_embeddings": context,
        "intermediate_size": intermediate,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": d,
        "vocab_size": vocab,
        "tie_word_embeddings": tied,
        "checkpoint_dtype": checkpoint_dtype,
        "qkv_bias": qkv_bias,
        "norms_per_layer": norms_per_layer,
        "qk_norm": qk_norm,
        "aliases": tuple(aliases),
        "sources": tuple(sources) if sources else (config_url,),
        "verified_on": VERIFIED_ON,
        "note": note,
    }


MODEL_DEFAULTS: Final[list[dict]] = [
    _profile(
        "Llama 7B", None, layers=32, hidden=4096, context=2048,
        intermediate=11008, heads=32, kv_heads=32, vocab=32000,
        checkpoint_dtype="float16", aliases=("LLaMA 7B", "Llama 1 7B", "LLaMA-1-7B"),
        sources=("https://arxiv.org/html/2302.13971v1", "https://arxiv.org/html/2307.09288v2"),
        note="Unversioned 'Llama 7B' explicitly means original LLaMA 1 here. Use 'Llama 2 7B' for Llama 2.",
    ),
    _profile(
        "Llama 2 7B", "meta-llama/Llama-2-7b-hf", layers=32, hidden=4096,
        context=4096, intermediate=11008, heads=32, kv_heads=32, vocab=32000,
        checkpoint_dtype="float16", aliases=("Llama-2-7B",),
        sources=("https://arxiv.org/html/2307.09288v2",),
    ),
    _profile(
        "Llama 3 8B", "meta-llama/Meta-Llama-3-8B", layers=32, hidden=4096,
        context=8192, intermediate=14336, heads=32, kv_heads=8, vocab=128256,
        aliases=("Llama-3-8B", "meta-llama/Llama-3-8B"),
        sources=(META_CONFIG_SOURCE, "https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md"),
    ),
    _profile(
        "Llama 3 70B", "meta-llama/Meta-Llama-3-70B", layers=80, hidden=8192,
        context=8192, intermediate=28672, heads=64, kv_heads=8, vocab=128256,
        aliases=("Llama-3-70B", "meta-llama/Llama-3-70B"),
        sources=(META_CONFIG_SOURCE, "https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md"),
    ),
    _profile(
        "Llama 3.1 8B", "meta-llama/Llama-3.1-8B", layers=32, hidden=4096,
        context=131072, intermediate=14336, heads=32, kv_heads=8, vocab=128256,
        aliases=("Llama-3.1-8B", "meta-llama/Meta-Llama-3.1-8B"),
        sources=(META_CONFIG_SOURCE, META_31_CARD),
    ),
    _profile(
        "Llama 3.1 70B", "meta-llama/Llama-3.1-70B", layers=80, hidden=8192,
        context=131072, intermediate=28672, heads=64, kv_heads=8, vocab=128256,
        aliases=("Llama-3.1-70B", "meta-llama/Meta-Llama-3.1-70B"),
        sources=(META_CONFIG_SOURCE, META_31_CARD),
    ),
    _profile(
        "Llama 3.2 1B", "meta-llama/Llama-3.2-1B", layers=16, hidden=2048,
        context=131072, intermediate=8192, heads=32, kv_heads=8, vocab=128256,
        tied=True, aliases=("Llama-3.2-1B",), sources=(META_CONFIG_SOURCE, META_32_CARD),
    ),
    _profile(
        "Llama 3.2 3B", "meta-llama/Llama-3.2-3B", layers=28, hidden=3072,
        context=131072, intermediate=8192, heads=24, kv_heads=8, vocab=128256,
        tied=True, aliases=("Llama-3.2-3B",), sources=(META_CONFIG_SOURCE, META_32_CARD),
    ),
    _profile(
        "Llama 3.3 70B", "meta-llama/Llama-3.3-70B-Instruct", layers=80,
        hidden=8192, context=131072, intermediate=28672, heads=64, kv_heads=8,
        vocab=128256, aliases=("Llama-3.3-70B", "Llama 3.3 70B Instruct"),
        sources=(META_CONFIG_SOURCE, META_33_CARD),
        note="The published Llama 3.3 70B checkpoint is instruction-tuned.",
    ),
    _profile(
        "Mistral 7B v0.3", "mistralai/Mistral-7B-v0.3", layers=32,
        hidden=4096, context=32768, intermediate=14336, heads=32, kv_heads=8,
        vocab=32768, aliases=("Mistral 7B", "Mistral-7B-v0.3"),
        note="Unversioned 'Mistral 7B' explicitly resolves to v0.3; v0.1 has a separate record.",
    ),
    _profile(
        "Mistral 7B v0.1", "mistralai/Mistral-7B-v0.1", layers=32,
        hidden=4096, context=32768, intermediate=14336, heads=32, kv_heads=8,
        vocab=32000, aliases=("Mistral-7B-v0.1",),
        note="Config maximum is 32768; sliding-window attention uses a 4096-token window.",
    ),
    _profile(
        "Qwen2.5 0.5B", "Qwen/Qwen2.5-0.5B", layers=24, hidden=896,
        context=32768, intermediate=4864, heads=14, kv_heads=2, vocab=151936,
        tied=True, qkv_bias=True,
    ),
    _profile(
        "Qwen2.5 1.5B", "Qwen/Qwen2.5-1.5B", layers=28, hidden=1536,
        context=131072, intermediate=8960, heads=12, kv_heads=2, vocab=151936,
        tied=True, qkv_bias=True,
        note="Base checkpoint config is 131072; do not substitute an Instruct checkpoint's config.",
    ),
    _profile(
        "Qwen2.5 3B", "Qwen/Qwen2.5-3B", layers=36, hidden=2048,
        context=32768, intermediate=11008, heads=16, kv_heads=2, vocab=151936,
        tied=True, qkv_bias=True,
    ),
    _profile(
        "Qwen2.5 7B", "Qwen/Qwen2.5-7B", layers=28, hidden=3584,
        context=131072, intermediate=18944, heads=28, kv_heads=4, vocab=152064,
        qkv_bias=True,
        note="Base checkpoint config is 131072; Instruct and extended-context variants are separate checkpoints.",
    ),
    _profile(
        "Qwen2.5 14B", "Qwen/Qwen2.5-14B", layers=48, hidden=5120,
        context=131072, intermediate=13824, heads=40, kv_heads=8, vocab=152064,
        qkv_bias=True,
    ),
    _profile(
        "Qwen2.5 32B", "Qwen/Qwen2.5-32B", layers=64, hidden=5120,
        context=131072, intermediate=27648, heads=40, kv_heads=8, vocab=152064,
        qkv_bias=True,
    ),
    _profile(
        "Qwen2.5-Coder 7B", "Qwen/Qwen2.5-Coder-7B", layers=28, hidden=3584,
        context=32768, intermediate=18944, heads=28, kv_heads=4, vocab=152064,
        qkv_bias=True,
        note="32768 is the unmodified config limit; a longer context requires the appropriate RoPE setup.",
    ),
    _profile(
        "Gemma 2 2B", "google/gemma-2-2b", layers=26, hidden=2304,
        context=8192, intermediate=9216, heads=8, kv_heads=4, vocab=256000,
        tied=True, head_dim=256, norms_per_layer=4,
        sources=(GEMMA_CONFIG_SOURCE, GEMMA_MODEL_SOURCE),
    ),
    _profile(
        "Gemma 2 9B", "google/gemma-2-9b", layers=42, hidden=3584,
        context=8192, intermediate=14336, heads=16, kv_heads=8, vocab=256000,
        tied=True, head_dim=256, norms_per_layer=4,
        sources=(GEMMA_CONFIG_SOURCE, GEMMA_MODEL_SOURCE),
    ),
    _profile(
        "TinyLlama 1.1B", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", layers=22,
        hidden=2048, context=2048, intermediate=5632, heads=32, kv_heads=4,
        vocab=32000, aliases=("TinyLlama-1.1B-Chat-v1.0",),
        note="Unversioned TinyLlama 1.1B explicitly selects Chat v1.0.",
    ),
    _profile(
        "Phi-3 mini 4k instruct", "microsoft/Phi-3-mini-4k-instruct", layers=32,
        hidden=3072, context=4096, intermediate=8192, heads=32, kv_heads=32,
        vocab=32064, aliases=("Phi-3-mini-4k-instruct",),
    ),
    # Additional dense models: compatible with the estimator's existing scope.
    _profile(
        "Qwen3 4B", "Qwen/Qwen3-4B", layers=36, hidden=2560,
        context=40960, intermediate=9728, heads=32, kv_heads=8, vocab=151936,
        tied=True, head_dim=128, qk_norm=True,
        sources=("https://huggingface.co/Qwen/Qwen3-4B/blob/main/config.json",
                 "https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py"),
    ),
    _profile(
        "Qwen3 8B", "Qwen/Qwen3-8B", layers=36, hidden=4096,
        context=40960, intermediate=12288, heads=32, kv_heads=8, vocab=151936,
        head_dim=128, qk_norm=True,
        sources=("https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json",
                 "https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py"),
    ),
    _profile(
        "Phi-4 mini instruct", "microsoft/Phi-4-mini-instruct", layers=32,
        hidden=3072, context=131072, intermediate=8192, heads=24, kv_heads=8,
        vocab=200064, tied=True, aliases=("Phi-4-mini-instruct",),
    ),
    _profile(
        "SmolLM2 1.7B", "HuggingFaceTB/SmolLM2-1.7B", layers=24,
        hidden=2048, context=8192, intermediate=8192, heads=32, kv_heads=32,
        vocab=49152, tied=True,
    ),
]


def normalize_model_name(value: str) -> str:
    """Case/space/hyphen/punctuation insensitive, without fuzzy size matching."""
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def get_model_defaults(model_name: str) -> dict | None:
    """Return a detached record for a registered display name or checkpoint ID.

    No suffix stripping or guessing: e.g. arbitrary fine-tuned, quantized, MoE,
    vision, and Instruct variants do not inherit a base config accidentally.
    Searches the small list directly, so adding a new entry takes effect at once.
    """
    if not isinstance(model_name, str):
        return None
    target = normalize_model_name(model_name)
    for profile in MODEL_DEFAULTS:
        names = (profile["model_name"], profile["checkpoint_id"], *profile["aliases"])
        if any(name is not None and normalize_model_name(name) == target for name in names):
            return deepcopy(profile)
    return None
