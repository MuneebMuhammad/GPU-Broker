"""Validation, state resolution, and deterministic GPU selection helpers."""

from decimal import Decimal, InvalidOperation
import json
import math
import re

from scripts.consts import (
    AGENT_FIELDS, ARCHITECTURE_FIELDS, TRAINING_ONLY_FIELDS, FIELD_QUESTIONS,
    INTEGER_FIELDS, MAX_FIELD_TEXT_LENGTH, ESTIMATOR_DEFAULTS, INFERENCE_DEFAULTS,
    DEFAULT_LORA_RANK, PRECISION_ALIASES, INFERENCE_PRECISION_ALIASES,
    METHOD_LABELS, PRECISION_LABELS, get_model_defaults, normalize_model_name,
)

def key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def positive(value, name: str, *, integer: bool = False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive {'integer' if integer else 'number'}.")
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive number.") from exc
    if not math.isfinite(number) or number <= 0 or (integer and not number.is_integer()):
        raise ValueError(f"{name} must be a finite positive {'integer' if integer else 'number'}.")
    return int(number) if integer else number


def parse_parameter_count(value: str | int | float) -> int:
    """Accept 7B, 400M, 100K, 15,700, 1.5B, or a plain positive integer."""
    if isinstance(value, bool):
        raise ValueError("parameter_count must be a positive integer or K/M/B string.")
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMB]?)\s*", str(value).replace(",", ""), re.I)
    if not match:
        raise ValueError("Invalid parameter_count. Examples: '7B', '400M', '100K', 15700.")
    try:
        count = Decimal(match[1]) * {"": 1, "K": 10**3, "M": 10**6, "B": 10**9}[match[2].upper()]
    except InvalidOperation as exc:
        raise ValueError("Invalid parameter_count.") from exc
    if not count.is_finite() or count <= 0 or count != count.to_integral_value():
        raise ValueError("parameter_count must resolve to a positive whole number.")
    return int(count)


def select_training_configuration(candidates, deadline=None):
    """Apply deadlines after estimating runtime for every memory-feasible candidate."""
    notes = []
    if deadline is not None and deadline != "urgent":
        deadline = positive(deadline, "training_time_constraint_hours")
        for candidate in candidates:
            meets = candidate["training_hours"] is not None and candidate["training_hours"] <= deadline
            candidate["meets_deadline"] = meets
            if candidate["feasible"] and not meets:
                candidate["feasible"] = False
                candidate["rejection_reasons"].append("Estimated training time exceeds the requested deadline.")
    feasible = [candidate for candidate in candidates if candidate["feasible"]]
    if not feasible:
        return None, notes
    if deadline == "urgent":
        # Urgency is one catalog step above the normal fit, not the fastest GPU.
        if len(feasible) > 1:
            notes.append("Urgent training: selected the next feasible catalog configuration.")
            return feasible[1], notes
        notes.append("Urgent training: no larger feasible configuration exists; retained the normal recommendation.")
    return feasible[0], notes


def checked_estimate(estimator, **values):
    """Do not emit Infinity/NaN JSON if finite inputs overflow during estimation."""
    try:
        result = estimator(**values)
        json.dumps(result, allow_nan=False)
    except (OverflowError, ValueError) as exc:
        raise ValueError("The supplied values cannot be estimated. Check their units and magnitude.") from exc
    return result


def validate_scalar(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("Field values must be strings or finite numbers.")
    if isinstance(value, str) and len(value) > MAX_FIELD_TEXT_LENGTH:
        raise ValueError("Field text is too long.")
    if not isinstance(value, str):
        try:
            if not math.isfinite(value):
                raise ValueError("Field values must be finite.")
        except OverflowError as exc:
            raise ValueError("Field number is too large.") from exc


def validate_state(state):
    if state is None:
        return {"values": {}, "unresolved_fields": [], "pending_fields": []}
    if not isinstance(state, dict) or set(state) != {"values", "unresolved_fields", "pending_fields"}:
        raise ValueError("state must contain values, unresolved_fields, and pending_fields.")
    values = state["values"]
    if not isinstance(values, dict) or not set(values).issubset(AGENT_FIELDS):
        raise ValueError("state.values contains unsupported fields.")
    for value in values.values():
        validate_scalar(value)
    for name in ("unresolved_fields", "pending_fields"):
        fields = state[name]
        if (not isinstance(fields, list) or len(fields) > len(AGENT_FIELDS)
                or any(not isinstance(f, str) or f not in AGENT_FIELDS for f in fields)
                or len(set(fields)) != len(fields)):
            raise ValueError(f"Invalid state.{name}.")
    if set(state["unresolved_fields"]) & values.keys():
        raise ValueError("An unresolved field cannot also have a value.")
    return {"values": dict(values), "unresolved_fields": list(state["unresolved_fields"]),
            "pending_fields": list(state["pending_fields"])}


def validate_updates(patch):
    if not isinstance(patch, dict) or set(patch) != {"reset", "updates"} or type(patch["reset"]) is not bool:
        raise ValueError("Invalid model response structure.")
    if not isinstance(patch["updates"], list) or len(patch["updates"]) > len(AGENT_FIELDS):
        raise ValueError("Invalid model updates.")
    seen = set()
    for item in patch["updates"]:
        if not isinstance(item, dict) or set(item) != {"field", "operation", "value"}:
            raise ValueError("Invalid field update.")
        field, operation, value = item["field"], item["operation"], item["value"]
        if not isinstance(field, str) or field not in AGENT_FIELDS or field in seen:
            raise ValueError("Unsupported or duplicate field update.")
        seen.add(field)
        if operation == "set":
            validate_scalar(value)
        elif operation not in ("clear", "unresolved") or value is not None:
            raise ValueError("Invalid field operation.")
    return patch


def model_identity(value):
    if not isinstance(value, str):
        return value
    profile = get_model_defaults(value)
    return normalize_model_name(profile["model_name"] if profile else value)


def merge_updates(state, patch):
    merged = validate_state(None if patch["reset"] else state)
    values, unresolved = merged["values"], set(merged["unresolved_fields"])
    updates = {item["field"]: item for item in patch["updates"]}
    model_update = updates.get("model_name")
    if model_update and model_identity(model_update["value"]) != model_identity(values.get("model_name")):
        # Never carry a previous model's custom architecture onto a new model.
        for field in ARCHITECTURE_FIELDS:
            values.pop(field, None)
            unresolved.discard(field)
    workload_update = updates.get("workload_type")
    if (workload_update and values.get("workload_type") in ("training", "inference")
            and workload_update["value"] in ("training", "inference")
            and workload_update["value"] != values.get("workload_type")):
        for field in TRAINING_ONLY_FIELDS:
            values.pop(field, None)
            unresolved.discard(field)
        # Do not turn inference INT4 weights into an explicit training dtype.
        mode = key(values.get("precision", ""))
        allowed = PRECISION_ALIASES if workload_update["value"] == "training" else INFERENCE_PRECISION_ALIASES
        if mode not in allowed:
            values.pop("precision", None)
            unresolved.discard("precision")
    for field, item in updates.items():
        values.pop(field, None)
        unresolved.discard(field)
        if item["operation"] == "set":
            values[field] = item["value"]
        elif item["operation"] == "unresolved":
            unresolved.add(field)
    merged["unresolved_fields"] = [field for field in AGENT_FIELDS if field in unresolved]
    merged["pending_fields"] = []
    return merged


def normalize_field(field, value):
    if field == "parameter_count":
        return parse_parameter_count(value)
    if field in ("model_name", "training_method", "precision", "workload_type"):
        if not isinstance(value, str) or not value.strip() or key(value) in ("null", "none"):
            raise ValueError("A nonempty value is required.")
        value = value.strip()
        if field == "training_method":
            if key(value) not in METHOD_LABELS:
                raise ValueError("Unsupported training method.")
            return METHOD_LABELS[key(value)]
        if field == "precision":
            return PRECISION_LABELS.get(key(value), value)
        if field == "workload_type":
            if value.lower() not in ("training", "inference"):
                raise ValueError("Specify training or inference.")
            return value.lower()
        return value
    if field == "training_time_constraint_hours" and isinstance(value, str) and value.lower().strip() == "urgent":
        return "urgent"
    return positive(value, field, integer=field in INTEGER_FIELDS)


def resolve_fields(state):
    fields = dict.fromkeys(AGENT_FIELDS)
    sources = dict.fromkeys(AGENT_FIELDS, "unset")
    issues = {field: "Please choose one value." for field in state["unresolved_fields"]}
    for field, value in state["values"].items():
        try:
            fields[field] = normalize_field(field, value)
            sources[field] = "user"
        except ValueError as exc:
            issues[field] = str(exc)
    profile = get_model_defaults(fields["model_name"])
    for field in (*ARCHITECTURE_FIELDS, "sequence_length"):
        if fields[field] is None and field not in issues and profile:
            fields[field] = profile[field]
            sources[field] = "model_registry"
    if profile is None and fields["parameter_count"] is None and "parameter_count" not in issues:
        # Only size tokens count; model versions and checkpoint '4k' context do not.
        match = re.search(r"(?:^|[\s_-])(\d+(?:\.\d+)?[MB])(?=$|[\s_-])", fields["model_name"] or "", re.I)
        if match:
            try:
                fields["parameter_count"] = parse_parameter_count(match[1])
                sources["parameter_count"] = "model_name"
            except ValueError:
                issues["parameter_count"] = "The model's size suffix is not a positive parameter count."
    workload = fields["workload_type"]
    defaults = {}
    inactive = []
    if workload == "training":
        defaults = {**ESTIMATOR_DEFAULTS}
    elif workload == "inference":
        defaults = {"batch_size": INFERENCE_DEFAULTS["concurrent_sequences"], "precision": INFERENCE_DEFAULTS["precision"]}
        inactive = list(TRAINING_ONLY_FIELDS)
    for field, value in defaults.items():
        if fields[field] is None and field not in issues:
            fields[field], sources[field] = value, "default"
    if workload == "training":
        if fields["training_method"] in ("LoRA", "QLoRA"):
            if fields["lora_rank"] is None and "lora_rank" not in issues:
                fields["lora_rank"], sources["lora_rank"] = DEFAULT_LORA_RANK, "default"
        elif fields["training_method"] is not None:
            inactive.append("lora_rank")
    for field in inactive:
        fields[field], sources[field] = None, "not_applicable"
        issues.pop(field, None)
    precision = fields["precision"]
    if precision is not None and workload is not None:
        allowed = PRECISION_ALIASES if workload == "training" else INFERENCE_PRECISION_ALIASES
        if key(precision) not in allowed:
            issues["precision"] = f"Precision is unsupported for {workload}."
        elif fields["training_method"] == "QLoRA" and PRECISION_ALIASES[key(precision)] != "bf16_mixed":
            issues["precision"] = "QLoRA requires BF16 or FP16 mixed compute."
    required = ["model_name", "workload_type"]
    if workload:
        required += ["parameter_count", "sequence_length"]
    if workload == "training":
        required.append("dataset_size_gb")
    for field in required:
        if fields[field] is None:
            issues.setdefault(field, "This field is required.")
    return fields, sources, issues


def clarification_message(issues):
    return "Please provide or clarify: " + "; ".join(
        f"{FIELD_QUESTIONS[field]} ({reason})" for field, reason in issues.items()
    ) + "."
