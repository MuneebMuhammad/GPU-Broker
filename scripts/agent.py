"""Stateless conversational extraction followed by deterministic GPU estimation."""

import json
import os

from openai import OpenAI, APIError, APITimeoutError
from dotenv import load_dotenv

load_dotenv(override=True)

from scripts.consts import (
    AGENT_MODEL, AGENT_PROMPT, AGENT_UPDATE_SCHEMA, AGENT_MAX_OUTPUT_TOKENS,
    AGENT_TIMEOUT_SECONDS, AGENT_MAX_RETRIES, MAX_MESSAGE_LENGTH,
    DEFAULT_LORA_RANK, INFERENCE_DEFAULTS,
)
from scripts.gpu_broker_estimator import estimate_training, estimate_inference
from scripts.utils import (
    validate_state, validate_updates, merge_updates, resolve_fields, clarification_message,
    checked_estimate,
)


class AgentUpstreamError(Exception):
    """Safe public error; API response bodies and credentials are never returned."""


class AgentTimeoutError(AgentUpstreamError):
    pass


class GPUAgent:
    def __init__(self, client=None):
        self.client = client if client is not None else OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            timeout=AGENT_TIMEOUT_SECONDS, max_retries=AGENT_MAX_RETRIES,
        )

    def respond(self, message, state=None):
        if not isinstance(message, str) or not message.strip() or len(message) > MAX_MESSAGE_LENGTH:
            raise ValueError(f"message must contain 1–{MAX_MESSAGE_LENGTH} characters.")
        state = validate_state(state)
        current, _, current_issues = resolve_fields(state)
        state["pending_fields"] = list(current_issues)
        try:
            response = self.client.responses.create(
                model=AGENT_MODEL, reasoning={"effort": "medium"},
                instructions=AGENT_PROMPT,
                input=json.dumps({"message": message, "state": state, "resolved_fields": current}, allow_nan=False),
                text={"format": {"type": "json_schema", "name": "gpu_field_updates",
                                 "strict": True, "schema": AGENT_UPDATE_SCHEMA}},
                max_output_tokens=AGENT_MAX_OUTPUT_TOKENS, store=False,
            )
        except APITimeoutError as exc:
            raise AgentTimeoutError("The AI service timed out. Retry with the same state.") from exc
        except APIError as exc:
            raise AgentUpstreamError("The AI service is unavailable. Retry with the same state.") from exc
        try:
            if response.status != "completed":
                raise ValueError("Incomplete model response.")
            patch = validate_updates(json.loads(response.output_text))
        except (ValueError, TypeError) as exc:
            raise AgentUpstreamError("The AI service returned an invalid response. Retry with the same state.") from exc

        state = merge_updates(state, patch)
        fields, sources, issues = resolve_fields(state)
        state["pending_fields"] = list(issues)
        result = {
            "status": "needs_input", "message": "", "state": state,
            "fields": fields, "field_sources": sources, "missing_fields": list(issues),
            "estimated_fields": [], "assumptions": [],
            "gpu_configuration": None, "estimated_time_hours": None,
        }
        if issues:
            result["message"] = clarification_message(issues)
            return result

        common = {field: fields[field] for field in (
            "model_name", "parameter_count", "sequence_length", "precision", "number_of_layers", "hidden_size",
        )}
        if fields["workload_type"] == "training":
            estimate = checked_estimate(estimate_training,
                **common, dataset_size_gb=fields["dataset_size_gb"], batch_size=fields["batch_size"],
                training_method=fields["training_method"], epochs=fields["epochs"],
                lora_rank=fields["lora_rank"] if fields["lora_rank"] is not None else DEFAULT_LORA_RANK,
                training_time_constraint_hours=fields["training_time_constraint_hours"],
            )
            # The heuristic h can be fractional. It is an estimate, never saved as user state.
            if estimate["hidden_size_inferred"]:
                fields["hidden_size"] = estimate["hidden_size_used"]
                sources["hidden_size"] = "estimated"
                result["estimated_fields"].append("hidden_size")
            result["assumptions"] = [
                "Dense-transformer training with checkpointing and ideal model-state sharding.",
                "Uncompressed text is estimated at 4 bytes per token; QLoRA time multiplier is 1.3.",
                "Training time excludes loading, checkpoint saves, and storage operations.",
            ]
            if estimate["activation_coefficient_source"] == "0.12*N**(2/3)":
                result["assumptions"].append("Activation memory uses a parameter-count heuristic because architecture is incomplete; no layer count is invented.")
        else:
            # Public batch_size is the number of sequences budgeted by the serving formula.
            estimate = checked_estimate(estimate_inference, **common, concurrent_sequences=fields["batch_size"])
            result["assumptions"] = [
                "Inference estimates memory only; runtime, latency, and throughput are not estimated.",
                f"KV-cache precision is fixed to {INFERENCE_DEFAULTS['kv_cache_precision']}; batch_size maps to concurrent_sequences.",
                "Urgency and numeric inference deadlines do not alter configuration selection.",
            ]
            if fields["number_of_layers"] is None or fields["hidden_size"] is None:
                result["assumptions"].append("KV-cache memory uses the parameter-count heuristic; missing layer count and hidden size remain unknown.")
        for field in ("number_of_layers", "hidden_size"):
            if fields[field] is None:
                sources[field] = "unavailable"
        result["assumptions"] += estimate.get("assumptions", []) + estimate["warnings"]
        if sources["sequence_length"] == "model_registry":
            result["assumptions"].append("Sequence length uses the registered checkpoint's full context limit; you can override it.")
        result["assumptions"] = list(dict.fromkeys(result["assumptions"]))
        result["gpu_configuration"] = estimate["recommended"]
        if estimate["recommended"] is None:
            result["status"] = "no_feasible_configuration"
            result["message"] = "No catalog configuration meets the requested memory, precision, and training-deadline constraints. You can revise the fields."
        else:
            config = estimate["recommended"]
            result["status"] = "ready"
            result["estimated_time_hours"] = config.get("training_hours")
            result["message"] = f"Recommended: {config['instance']} ({config['gpu_count']} × {config['gpu']}); estimated memory {config['required_memory_gb_per_gpu']:.2f} GB per GPU."
            if result["estimated_time_hours"] is not None:
                result["message"] += f" Estimated training time: {result['estimated_time_hours']:.3g} hours."
            else:
                result["message"] += " Inference runtime is not estimated."
        defaults = [field for field, source in sources.items() if source == "default"]
        if defaults:
            result["message"] += " Defaults used: " + ", ".join(f"{field}={fields[field]}" for field in defaults) + "."
        if result["estimated_fields"]:
            result["message"] += " Estimated fields: " + ", ".join(result["estimated_fields"]) + "."
        if fields["number_of_layers"] is None or sources["hidden_size"] in ("estimated", "unavailable"):
            result["message"] += " Memory uses architecture heuristics; see assumptions."
        return result
