"""Deterministic dense-transformer training and inference resource estimates.

Python 3.10+, standard library only. No network access, model download, or training.
All GB values are decimal (1 GB = 10**9 bytes). Training assumes checkpointing.
Run python -m scripts.gpu_broker_estimator from the repository root for an example.
"""

from __future__ import annotations

import json
import math

from scripts.consts import (
    ESTIMATOR_DEFAULTS, get_model_defaults, normalize_model_name,
    INFERENCE_DEFAULTS, INFERENCE_WEIGHT_BYTES, INFERENCE_PRECISION_ALIASES,
    GPU_CATALOG, DEMO_HOURLY_PRICES_USD, TRAINING_METHOD,
    PRECISION_ALIASES, FULL_FT_STATE_BYTES_PER_PARAMETER, DEFAULT_LORA_RANK,
    TRAINING_SELECTION_RULE, INFERENCE_SELECTION_RULE,
)
from scripts.utils import key as _key, positive as _positive, parse_parameter_count, select_training_configuration


def estimate_training(
    model_name: str,
    parameter_count: str | int | float | None = None,
    dataset_size_gb: float | None = None,
    sequence_length: int | None = None,
    batch_size: int | None = None,
    training_method: str | None = None,
    precision: str | None = None,
    number_of_layers: int | None = None,
    hidden_size: int | None = None,
    epochs: float | None = None,
    *,
    lora_rank: int = DEFAULT_LORA_RANK,
    training_time_constraint_hours: float | str | None = None,
    bytes_per_token: float = 4.0,
    tokens_per_epoch: int | None = None,
    shard_model_states: bool = True,
    extra_memory_gb_per_gpu: float = 0.0,
    qlora_time_multiplier: float = 1.3,
) -> dict:
    """Estimate all five configurations and choose the first feasible one.

    Missing (None or omitted) architecture inputs are filled from consts.py:
        parameter_count, number_of_layers, hidden_size, and sequence_length.
        Explicit values always win, even if they differ from the registry.
        Default sequence length is the checkpoint's CONFIGURED CONTEXT LIMIT;
        pass the actual training length (e.g. 2048/4096) for your workload.
    Missing batch_size, training_method, precision, and epochs are filled from
        ESTIMATOR_DEFAULTS. These are workflow defaults, not model properties.
        Dataset size is still required unless tokens_per_epoch is supplied.
    Unknown/custom models need explicit parameter_count and sequence_length.
        Missing layer count/hidden size keep the original estimator heuristics.
        Null/blank model names and invalid numeric values are rejected.

    Selection order: ml.g5.2xlarge, ml.g6e.2xlarge, ml.g6e.12xlarge,
        ml.p4d.24xlarge, ml.p5.48xlarge. Select the first configuration with
        enough per-GPU memory and support for the requested precision.
        A numeric training_time_constraint_hours also constrains selection.
        "urgent" selects the next feasible configuration, or retains the only one.
        Prices are illustrative and do not determine selection.

    batch_size: PER-GPU micro-batch size, before gradient accumulation. It stays
        fixed on every candidate, so global batch size can change with GPU count.
    dataset_size_gb: uncompressed training text bytes / 1e9; NOT archive size,
        image data, or automatically the on-disk size of Parquet/JSON files.
    epochs: defaults to 1. D = tokens_per_epoch * epochs. An exact token count
        overrides the bytes_per_token conversion when supplied.
    precision aliases:
        BF16 / FP16 / bf16_mixed -> 16 bytes/parameter for full fine-tuning;
        FP32 / pure_fp32 -> 16;
        pure_bf16 / pure_fp16 -> 8 (including LOW-PRECISION Adam states);
        bf16_8bit_adam / BF16 mixed + 8-bit Adam -> 10;
        FP8 / fp8_mixed -> 16, with BF16 activations/frozen weights.
        Eight bytes is the user's explicit low-precision optimizer assumption,
        NOT a universal memory requirement of BF16/FP16 training.
    Pretraining uses the full-weight training formulas. Fractional epochs are allowed.
    LoRA/QLoRA: always 16 bytes per adapter parameter per the supplied formula.
        Quantized base storage in QLoRA is always 0.52 bytes/parameter. QLoRA
        accepts BF16/FP16 mixed only; it does NOT use FP8 compute throughput.
    sharding: when True, full-FT states or LoRA/QLoRA base weights are ideally
        sharded across all GPUs. Adapter states and activations are replicated.
        Requires a compatible distributed implementation, including FSDP-QLoRA
        for quantized sharding. False means ordinary replicated data parallelism.
    missing architecture after registry lookup: C = .12*N**(2/3) if either
        L or h is still missing, exactly
        as requested (do NOT divide by 10 again). Missing h for P and S/h is
        approximated from N = 12*L*h**2; if L is absent too, assume L = h/128.
        These are crude dense-transformer heuristics for unregistered models.
    pricing: uses your supplied DEMO_HOURLY_PRICES_USD constant.
    time: effective TFLOPS already includes GPU count and MFU. Do not multiply
        by either again. QLoRA divides throughput by qlora_time_multiplier.

    Returns a JSON-serializable dict, including per-GPU memory breakdown,
    elapsed training hours, GPU-hours, cost, feasibility, and recommendation.
    Model loading, data loading, checkpoint saves, and storage costs are omitted.
    Estimates are conditional on these heuristics; they are not fit guarantees.
    """
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a nonempty string.")
    if normalize_model_name(model_name) in {"", "null", "none"}:
        raise ValueError("model_name cannot be blank, null, or None.")

    profile = get_model_defaults(model_name)
    defaulted_fields = []
    input_sources = {}
    resolved = {
        "parameter_count": parameter_count,
        "number_of_layers": number_of_layers,
        "hidden_size": hidden_size,
        "sequence_length": sequence_length,
    }
    for field, value in resolved.items():
        if value is None and profile is not None:
            resolved[field] = profile[field]
            defaulted_fields.append(field)
            input_sources[field] = "model_defaults"
        else:
            input_sources[field] = "explicit" if value is not None else "heuristic"

    missing_required = [field for field in ("parameter_count", "sequence_length") if resolved[field] is None]
    if missing_required:
        raise ValueError(
            f"No registered defaults for {model_name!r}. Provide "
            + ", ".join(missing_required)
            + "; number_of_layers and hidden_size can also be supplied."
        )
    parameter_count = resolved["parameter_count"]
    sequence_length = resolved["sequence_length"]
    number_of_layers = resolved["number_of_layers"]
    hidden_size = resolved["hidden_size"]
    workflow = {"batch_size": batch_size, "training_method": training_method,
                "precision": precision, "epochs": epochs}
    for field, value in workflow.items():
        if value is None:
            workflow[field] = ESTIMATOR_DEFAULTS[field]
            defaulted_fields.append(field)
            input_sources[field] = "estimator_defaults"
        else:
            input_sources[field] = "explicit"
    batch_size = workflow["batch_size"]
    training_method = workflow["training_method"]
    precision = workflow["precision"]
    epochs = workflow["epochs"]

    n = parse_parameter_count(parameter_count)
    if dataset_size_gb is None and tokens_per_epoch is None:
        raise ValueError("Provide dataset_size_gb or tokens_per_epoch; dataset size is not a model property.")
    size = None if dataset_size_gb is None else _positive(dataset_size_gb, "dataset_size_gb")
    s = _positive(sequence_length, "sequence_length", integer=True)
    batch = _positive(batch_size, "batch_size", integer=True)
    epochs = _positive(epochs, "epochs")
    rank = _positive(lora_rank, "lora_rank", integer=True)
    bpt = _positive(bytes_per_token, "bytes_per_token")
    slowdown = _positive(qlora_time_multiplier, "qlora_time_multiplier")
    layers = None if number_of_layers is None else _positive(number_of_layers, "number_of_layers", integer=True)
    supplied_h = None if hidden_size is None else _positive(hidden_size, "hidden_size", integer=True)
    if not isinstance(shard_model_states, bool):
        raise ValueError("shard_model_states must be True or False.")
    if isinstance(extra_memory_gb_per_gpu, bool):
        raise ValueError("extra_memory_gb_per_gpu must be finite and nonnegative.")
    extra = float(extra_memory_gb_per_gpu)
    if not math.isfinite(extra) or extra < 0:
        raise ValueError("extra_memory_gb_per_gpu must be finite and nonnegative.")

    method = TRAINING_METHOD.get(_key(training_method))
    if method is None:
        raise ValueError("training_method must be LoRA, QLoRA, Full fine-tuning, or Pretraining.")

    mode = PRECISION_ALIASES.get(_key(precision))
    if mode is None:
        raise ValueError("Unknown precision. Use BF16, FP32, pure_bf16, pure_fp16, bf16_8bit_adam, or FP8.")
    if method == "qlora" and mode != "bf16_mixed":
        raise ValueError("This QLoRA formula requires BF16/FP16 mixed compute; use precision='BF16'.")

    warnings = []
    if profile is not None and profile["note"]:
        warnings.append(profile["note"])
    if input_sources["sequence_length"] == "model_defaults":
        warnings.append("Sequence length defaulted to the checkpoint context limit; override it with the actual training length if shorter.")
    if profile is not None and s > profile["max_position_embeddings"]:
        warnings.append("Explicit sequence_length exceeds the registered checkpoint context limit; the estimator does not configure context extension.")
    if profile is None:
        warnings.append("Unregistered/custom model: using explicit inputs and any remaining architecture heuristics.")
    h = supplied_h
    if h is None:
        h = math.sqrt(n / (12 * layers)) if layers is not None else (128 * n / 12) ** (1 / 3)
        warnings.append("Hidden size inferred.")
    if layers is not None and supplied_h is not None:
        c = 34 * layers * supplied_h / 10
        c_source = "34*L*h/10"
    else:
        c = 0.12 * n ** (2 / 3)
        c_source = "0.12*N**(2/3)"

    if tokens_per_epoch is None:
        epoch_tokens = size * 1e9 / bpt
        warnings.append("Token count estimated from uncompressed text size and bytes_per_token.")
    else:
        epoch_tokens = _positive(tokens_per_epoch, "tokens_per_epoch", integer=True)
    d = epoch_tokens * epochs
    t = batch * s
    activation_bytes = 4 if mode == "pure_fp32" else 2
    full_state_bytes = FULL_FT_STATE_BYTES_PER_PARAMETER[mode]
    adapter_params = 0 if method == "full_ft" else 1.6 * rank * n / h
    activation_gb = t * c * (activation_bytes / 2) / 1e9
    replicated_adapter_gb = 16 * adapter_params / 1e9
    base_bytes = full_state_bytes if method == "full_ft" else (0.52 if method == "qlora" else activation_bytes)
    flops = n * d * ((8 if method == "full_ft" else 6) + s / h)

    candidates = []
    for name, gpu, gpus, vram, bf16_tf, fp32_tf, fp8_tf in GPU_CATALOG:
        shards = gpus if shard_model_states else 1
        fixed_gb = base_bytes * n / shards / 1e9
        required_gb = fixed_gb + replicated_adapter_gb + activation_gb + extra
        usable_gb = 0.9 * vram
        effective_tf = fp32_tf if mode == "pure_fp32" else (fp8_tf if mode == "fp8_mixed" else bf16_tf)
        supported = effective_tf is not None
        fits = required_gb <= usable_gb
        feasible = supported and fits
        hours = None
        if feasible:
            hours = flops / (effective_tf * 1e12 * 3600)
            if method == "qlora":
                hours *= slowdown
        reasons = []
        if not supported:
            reasons.append("No native FP8 support in the supplied hardware table.")
        if not fits:
            reasons.append(f"Needs {required_gb:.3f} GB/GPU; only {usable_gb:.3f} GB/GPU usable.")
        candidates.append({
            "instance": name, "gpu": gpu, "gpu_count": gpus, "shards": shards,
            "usable_memory_gb_per_gpu": usable_gb, "usable_memory_gb_total": usable_gb * gpus,
            "fixed_base_or_full_states_gb_per_gpu": fixed_gb,
            "adapter_states_gb_per_gpu": replicated_adapter_gb,
            "activation_memory_gb_per_gpu": activation_gb,
            "extra_memory_gb_per_gpu": extra, "required_memory_gb_per_gpu": required_gb,
            "required_memory_gb_total": required_gb * gpus,
            "effective_tflops_per_instance": effective_tf,
            "feasible": feasible, "rejection_reasons": reasons,
            "training_hours": hours, "gpu_hours": None if hours is None else hours * gpus,
            "hourly_price_usd": DEMO_HOURLY_PRICES_USD[name],
            "estimated_cost_usd": None if hours is None else hours * DEMO_HOURLY_PRICES_USD[name],
        })
    best, selection_notes = select_training_configuration(candidates, training_time_constraint_hours)
    warnings.extend(selection_notes)
    return {
        "model_name": model_name, "parameter_count": n, "training_method": method,
        "resolved_model_name": None if profile is None else profile["model_name"],
        "checkpoint_id": None if profile is None else profile["checkpoint_id"],
        "defaulted_fields": defaulted_fields, "input_sources": input_sources,
        "model_default_sources": [] if profile is None else list(profile["sources"]),
        "sequence_length": s, "number_of_layers_used": layers,
        "precision": mode, "epochs": epochs, "checkpointing_enabled": True,
        "micro_batch_size_per_gpu": batch,
        "tokens_per_epoch": epoch_tokens, "total_training_tokens": d,
        "hidden_size_used": h, "hidden_size_inferred": supplied_h is None,
        "activation_coefficient_c": c, "activation_coefficient_source": c_source,
        "estimated_adapter_parameters": adapter_params,
        "full_ft_state_bytes_per_parameter": full_state_bytes if method == "full_ft" else None,
        "estimated_flops": flops,
        "candidates": candidates, "recommended": best,
        "status": "estimated_configuration_found" if best else "no_feasible_configuration",
        "selection_rule": TRAINING_SELECTION_RULE,
        "training_time_constraint_hours": training_time_constraint_hours,
        "prices_are_illustrative": True,
        "warnings": warnings,
    }


def estimate_inference(
    model_name: str,
    parameter_count: str | int | float | None = None,
    sequence_length: int | None = None,
    precision: str | None = None,
    number_of_layers: int | None = None,
    hidden_size: int | None = None,
    concurrent_sequences: int | None = None,
    *,
    kv_cache_precision: str | None = None,
    shard_model_states: bool = True,
    extra_memory_gb_per_gpu: float = 0.0,
) -> dict:
    """Estimate inference memory and select the first feasible GPU configuration.

    Known L and h: M_bytes = 1.2*N*b_w + B*S*2*L*h*b_kv.
    Missing either: M_bytes = 1.2*N*b_w + B*S*0.14*N**(2/3)*(b_kv/2).
    The fallback is exactly the supplied 0.14 formula for the default BF16 KV
    cache. FP8 KV storage halves that BF16 fallback coefficient.

    N, L, h, and sequence_length are filled from consts.py when omitted/None.
    Explicit inputs always win. Unknown models need N and sequence_length;
    if L or h is unavailable, use the fallback without inventing architecture.
    B defaults to 1; weight precision and KV precision independently default
    to BF16. Weight formats: PureFP32, BF16, FP16, FP8, INT8, INT4.

    S is the SAME sequence_length argument/default as in estimate_training.
    For inference it budgets the total prompt-plus-generated context per
    concurrent sequence. An omitted S uses the checkpoint's full context limit.

    For shard_model_states=True, total model memory AND KV cache are evenly
    sharded across all GPUs on each candidate. B applies to that sharded group.
    This is an ideal model/tensor-parallel assumption, not data parallelism.
    False keeps the entire model and B sequences' KV cache on EACH GPU; B is
    then per replica. Neither mode assumes CPU offloading or additional nodes.

    The supplied formula uses full hidden size, without a GQA reduction, even
    when the registry knows the number of KV heads. No checkpointing/optimizer
    state is included. 20% weight overhead is part of the requested formula;
    90% usable VRAM and optional extra_memory_gb_per_gpu are also retained.

    FP8 weights or FP8 KV require native FP8 support in the supplied GPU table.
    INT8/INT4 are storage estimates and assume suitable serving kernels.
    This estimates memory feasibility only, not latency, throughput, deployment,
    serving-runtime compatibility, or endpoint pricing.
    """
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a nonempty string.")
    if normalize_model_name(model_name) in {"", "null", "none"}:
        raise ValueError("model_name cannot be blank, null, or None.")

    profile = get_model_defaults(model_name)
    defaulted_fields = []
    input_sources = {}
    resolved = {
        "parameter_count": parameter_count,
        "sequence_length": sequence_length,
        "number_of_layers": number_of_layers,
        "hidden_size": hidden_size,
    }
    for field, value in resolved.items():
        if value is None and profile is not None:
            resolved[field] = profile[field]
            defaulted_fields.append(field)
            input_sources[field] = "model_defaults"
        else:
            input_sources[field] = "explicit" if value is not None else "unavailable"
    missing = [field for field in ("parameter_count", "sequence_length") if resolved[field] is None]
    if missing:
        raise ValueError(f"No registered defaults for {model_name!r}. Provide " + ", ".join(missing) + ".")

    workflow = {"concurrent_sequences": concurrent_sequences,
                "precision": precision, "kv_cache_precision": kv_cache_precision}
    for field, value in workflow.items():
        if value is None:
            workflow[field] = INFERENCE_DEFAULTS[field]
            defaulted_fields.append(field)
            input_sources[field] = "inference_defaults"
        else:
            input_sources[field] = "explicit"

    n = parse_parameter_count(resolved["parameter_count"])
    s = _positive(resolved["sequence_length"], "sequence_length", integer=True)
    b = _positive(workflow["concurrent_sequences"], "concurrent_sequences", integer=True)
    layers = None if resolved["number_of_layers"] is None else _positive(resolved["number_of_layers"], "number_of_layers", integer=True)
    h = None if resolved["hidden_size"] is None else _positive(resolved["hidden_size"], "hidden_size", integer=True)
    mode = INFERENCE_PRECISION_ALIASES.get(_key(workflow["precision"]))
    if mode is None:
        raise ValueError("Inference precision must be PureFP32, BF16, FP16, FP8, INT8, or INT4.")
    kv_mode = INFERENCE_PRECISION_ALIASES.get(_key(workflow["kv_cache_precision"]))
    if kv_mode is None:
        raise ValueError("kv_cache_precision must be PureFP32, BF16, FP16, FP8, INT8, or INT4.")
    if not isinstance(shard_model_states, bool):
        raise ValueError("shard_model_states must be True or False.")
    try:
        extra = float(extra_memory_gb_per_gpu)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("extra_memory_gb_per_gpu must be finite and nonnegative.") from exc
    if isinstance(extra_memory_gb_per_gpu, bool) or not math.isfinite(extra) or extra < 0:
        raise ValueError("extra_memory_gb_per_gpu must be finite and nonnegative.")

    b_w = INFERENCE_WEIGHT_BYTES[mode]
    b_kv = INFERENCE_WEIGHT_BYTES[kv_mode]
    raw_weights_gb = n * b_w / 1e9
    model_memory_gb = 1.2 * n * b_w / 1e9
    if layers is not None and h is not None:
        kv_bytes_per_token = 2 * layers * h * b_kv
        kv_formula = "2*L*h*b_kv"
    else:
        kv_bytes_per_token = 0.14 * n ** (2 / 3) * (b_kv / 2)
        kv_formula = "0.14*N**(2/3)*(b_kv/2)"
    kv_memory_gb = b * s * kv_bytes_per_token / 1e9
    memory_gb = model_memory_gb + kv_memory_gb
    warnings = ["KV cache uses the full-hidden-size formula requested, without a grouped-query attention reduction."]
    if profile is not None and profile["note"]:
        warnings.append(profile["note"])
    if input_sources["sequence_length"] == "model_defaults":
        warnings.append("Sequence length defaulted to the checkpoint context limit; pass your serving context length if shorter.")
    if profile is not None and s > profile["max_position_embeddings"]:
        warnings.append("Explicit sequence_length exceeds the registered checkpoint context limit; context extension is not configured.")
    if layers is None or h is None:
        warnings.append("Layers or hidden size unavailable; using the parameter-count KV cache heuristic.")

    candidates = []
    for name, gpu, gpus, vram, _bf16_tf, _fp32_tf, fp8_tf in GPU_CATALOG:
        shards = gpus if shard_model_states else 1
        model_per_gpu = model_memory_gb / shards
        kv_per_gpu = kv_memory_gb / shards
        required_gb = model_per_gpu + kv_per_gpu + extra
        usable_gb = 0.9 * vram
        supported = not ((mode == "fp8" or kv_mode == "fp8") and fp8_tf is None)
        fits = required_gb <= usable_gb
        reasons = []
        if not supported:
            reasons.append("Requested FP8 weights or KV cache require native FP8 support in the supplied GPU table.")
        if not fits:
            reasons.append(f"Needs {required_gb:.3f} GB/GPU; only {usable_gb:.3f} GB/GPU usable.")
        candidates.append({
            "instance": name, "gpu": gpu, "gpu_count": gpus, "shards": shards,
            "usable_memory_gb_per_gpu": usable_gb,
            "usable_memory_gb_total": usable_gb * gpus,
            "model_memory_gb_per_gpu": model_per_gpu,
            "kv_cache_memory_gb_per_gpu": kv_per_gpu,
            "extra_memory_gb_per_gpu": extra,
            "required_memory_gb_per_gpu": required_gb,
            "required_memory_gb_total": required_gb * gpus,
            "precision_supported": supported, "fits_memory": fits,
            "feasible": supported and fits, "rejection_reasons": reasons,
        })
    best = next((row for row in candidates if row["feasible"]), None)
    return {
        "model_name": model_name,
        "resolved_model_name": None if profile is None else profile["model_name"],
        "checkpoint_id": None if profile is None else profile["checkpoint_id"],
        "parameter_count": n, "sequence_length": s, "concurrent_sequences": b,
        "number_of_layers_used": layers, "hidden_size_used": h,
        "precision": mode, "bytes_per_weight": b_w,
        "kv_cache_precision": kv_mode, "bytes_per_kv_element": b_kv,
        "raw_weight_memory_gb": raw_weights_gb,
        "model_memory_gb_including_overhead": model_memory_gb,
        "kv_cache_bytes_per_token": kv_bytes_per_token,
        "kv_cache_memory_gb": kv_memory_gb,
        "estimated_memory_gb": memory_gb,
        "memory_formula": "1.2*N*b_w + B*S*kv_cache_bytes_per_token",
        "kv_cache_formula": kv_formula,
        "shard_model_states": shard_model_states,
        "memory_distribution": "even model-and-KV sharding" if shard_model_states else "replicated model and B-sequence KV cache on each GPU",
        "defaulted_fields": defaulted_fields, "input_sources": input_sources,
        "model_default_sources": [] if profile is None else list(profile["sources"]),
        "candidates": candidates, "recommended": best,
        "status": "estimated_configuration_found" if best else "no_feasible_configuration",
        "selection_rule": INFERENCE_SELECTION_RULE,
        "assumptions": [
            "Multi-GPU sharding, if enabled, is idealized and requires a compatible inference runtime.",
        ],
        "warnings": warnings,
    }


if __name__ == "__main__":
    # result = estimate_training(
    #     model_name="Llama 3.1 8B", dataset_size_gb=0.4,
    #     sequence_length=4096, batch_size=2, training_method="LoRA",
    #     # Parameter count, layers, and hidden size come from consts.py.
    # )
    # print(json.dumps(result, indent=2, allow_nan=False))

    result = estimate_inference(
    model_name="Llama 3.1 8B",
    sequence_length=4096,
    precision="BF16",
    concurrent_sequences=1,     # Default: 1
    kv_cache_precision="BF16",  # Default: BF16
    )

    print(result["estimated_memory_gb"])
    print(result["recommended"])
