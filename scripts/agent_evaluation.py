"""Development evidence for the exact application agent, without changing its API."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from uuid import uuid4

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from scripts.agent import GPUAgent, AgentUpstreamError
from scripts.consts import (
    AGENT_PROMPT, AGENT_UPDATE_SCHEMA, AGENT_MODEL, AGENT_MAX_OUTPUT_TOKENS,
    AGENT_TIMEOUT_SECONDS, AGENT_MAX_RETRIES,
)
from scripts.utils import parse_parameter_count, validate_state, validate_updates


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "data/agent_development_cases.json"
UPDATE_VALIDATOR = Draft202012Validator(AGENT_UPDATE_SCHEMA)


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def load_agent_cases(path=CASES_PATH):
    suite = json.loads(Path(path).read_text())
    assert suite["split"] == "development"
    ids = [c["conversation_id"] for c in suite["conversations"]]
    assert ids and len(ids) == len(set(ids))
    for conversation in suite["conversations"]:
        validate_state(conversation["initial_state"])
        assert conversation["turns"]
        for number, turn in enumerate(conversation["turns"], 1):
            assert turn["turn"] == number and turn["message"].strip()
            UPDATE_VALIDATOR.validate(turn["expected_update"])
            validate_updates(turn["expected_update"])
            validate_state(turn["expected_state"])
    return suite


def _field_value(field, value):
    # Both numeric counts and K/M/B strings are expressly allowed by AGENT_PROMPT.
    if field == "parameter_count" and value is not None:
        try:
            return parse_parameter_count(value)
        except ValueError:
            return value  # Preserve invalid user facts; the application asks for correction.
    return value


def canonical_update(update):
    UPDATE_VALIDATOR.validate(update)
    validate_updates(update)
    return {
        "is_llm_request": update["is_llm_request"], "reset": update["reset"],
        "updates": {
            item["field"]: {
                "operation": item["operation"],
                "value": _field_value(item["field"], item["value"]),
            } for item in update["updates"]
        },
    }


def canonical_state(state):
    state = validate_state(state)
    return {
        "values": {k: _field_value(k, v) for k, v in state["values"].items()},
        "unresolved_fields": sorted(state["unresolved_fields"]),
        "pending_fields": sorted(state["pending_fields"]),
    }


def equivalent(expected, actual, field=None):
    """Exact keys/values, numeric equivalence, and 1e-9 GB conversion tolerance."""
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and expected.keys() == actual.keys()
                and all(equivalent(v, actual[k], field if k == "value" else k)
                        for k, v in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(expected) == len(actual)
                and all(equivalent(e, a, field) for e, a in zip(expected, actual)))
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(expected) is type(actual) and expected == actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return (math.isfinite(actual) and (
            math.isclose(expected, actual, abs_tol=1e-9, rel_tol=0)
            if field == "dataset_size_gb" else expected == actual))
    return expected == actual


class _RecordingResponses:
    """Forward the application's unchanged request; retain its response before merge."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def create(self, **kwargs):
        call = {"request": deepcopy(kwargs), "started_at_utc": datetime.now(timezone.utc).isoformat()}
        self.calls.append(call)
        start = time.perf_counter()
        try:
            response = self.responses.create(**kwargs)
        finally:
            call["sdk_call_s"] = time.perf_counter() - start
        call.update(
            response_id=response.id, response_model=response.model,
            response_status=response.status, raw_output=response.output_text,
            usage=response.usage.model_dump(mode="json") if response.usage else None,
        )
        return response


class _RecordingClient:
    def __init__(self, client):
        self.responses = _RecordingResponses(client.responses)


def score_agent_turn(turn, call, result, actual_state):
    raw_update, valid, update_correct = None, False, False
    try:
        raw_update = json.loads(call.get("raw_output", ""))
        actual_update = canonical_update(raw_update)
        valid = call.get("response_status") == "completed"
        update_correct = valid and equivalent(canonical_update(turn["expected_update"]), actual_update)
    except (ValueError, TypeError, ValidationError):
        pass
    state_correct = result is not None and equivalent(
        canonical_state(turn["expected_state"]), canonical_state(actual_state))
    resolved_checks = {}
    for expected_key, result_key in (
        ("expected_resolved_fields", "fields"), ("expected_field_sources", "field_sources"),
    ):
        resolved_checks[result_key] = {
            field: bool(result is not None and field in result[result_key]
                        and equivalent(value, result[result_key][field], field))
            for field, value in turn[expected_key].items()
        }
    resolved_correct = result is not None and all(
        ok for checks in resolved_checks.values() for ok in checks.values())
    flags = {
        "valid_update": valid, "update_correct": update_correct,
        "state_correct": state_correct, "resolved_checks_correct": resolved_correct,
    }
    flags["turn_pass"] = all(flags.values())
    return raw_update, flags, resolved_checks


def run_agent_development(suite=None, *, client=None, output_dir=None):
    """Run sequential conversations. An injected client is always labeled simulated.

    Only actual returned state reaches the next turn. On an application error,
    retain the previous state as the CLI does, score the failure, and continue.
    Gold labels are never sent to the model. Each completed turn is flushed to disk.
    """
    suite = deepcopy(suite if suite is not None else load_agent_cases())
    mode = "live" if client is None else "simulated"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    destination = Path(output_dir or ROOT / "data/agent_evaluation_runs") / run_id
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": run_id, "mode": mode, "suite_id": suite["suite_id"], "split": "development",
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "suite_sha256": _sha(_json(suite).encode()),
        "prompt_sha256": _sha(AGENT_PROMPT.encode()),
        "schema_sha256": _sha(_json(AGENT_UPDATE_SCHEMA).encode()),
        "model": AGENT_MODEL, "reasoning": {"effort": "medium"},
        "max_output_tokens": AGENT_MAX_OUTPUT_TOKENS, "store": False,
        "sdk_timeout_s": AGENT_TIMEOUT_SECONDS, "sdk_max_retries": AGENT_MAX_RETRIES,
        "local_response_cache": False, "max_workers": 1,
        "planned_conversations": len(suite["conversations"]),
        "planned_turns": sum(len(c["turns"]) for c in suite["conversations"]),
        "source_sha256": {name: _sha((ROOT / name).read_bytes()) for name in (
            "scripts/agent.py", "scripts/consts.py", "scripts/utils.py",
            "scripts/gpu_broker_estimator.py", "scripts/agent_evaluation.py",
        )},
    }
    (destination / "manifest.json").write_text(_json(manifest) + "\n")
    (destination / "expected_cases.json").write_text(_json(suite) + "\n")
    agent = GPUAgent(client=client)
    agent.client = _RecordingClient(agent.client)
    records = []
    with (destination / "turns.jsonl").open("w") as evidence:
        for conversation in suite["conversations"]:
            actual_state = deepcopy(conversation["initial_state"])
            expected_prior = deepcopy(conversation["initial_state"])
            for turn in conversation["turns"]:
                incoming = deepcopy(actual_state)
                result, error = None, None
                agent.client.responses.calls.clear()
                try:
                    result = agent.respond(turn["message"], deepcopy(incoming))
                    actual_state = deepcopy(result["state"])
                except (AgentUpstreamError, ValueError) as exc:
                    error = type(exc).__name__ + ": " + str(exc)
                calls = agent.client.responses.calls
                assert len(calls) <= 1, "Unexpected additional logical API calls"
                call = calls[0] if calls else {}
                raw_update, flags, checks = score_agent_turn(turn, call, result, actual_state)
                record = {
                    "run_id": run_id, "conversation_id": conversation["conversation_id"],
                    "purpose": conversation["purpose"], **deepcopy(turn),
                    "expected_incoming_state": expected_prior, "actual_incoming_state": incoming,
                    "actual_update": raw_update, "actual_state": deepcopy(actual_state),
                    "application_response": result, "api_call": call, "error": error,
                    "scores": flags, "resolved_checks": checks,
                    "mismatches": [name for name, passed in flags.items() if not passed and name != "turn_pass"],
                }
                records.append(record)
                evidence.write(_json(record) + "\n")
                evidence.flush()
                expected_prior = deepcopy(turn["expected_state"])
    totals = {name: sum(r["scores"][name] for r in records) for name in (
        "valid_update", "update_correct", "state_correct", "resolved_checks_correct", "turn_pass",
    )}
    conversation_results = {
        c["conversation_id"]: all(r["scores"]["turn_pass"] for r in records
                                  if r["conversation_id"] == c["conversation_id"])
        for c in suite["conversations"]
    }
    summary = {
        "mode": mode, "turns": len(records), **totals,
        "conversations": len(conversation_results),
        "conversations_correct": sum(conversation_results.values()),
        "conversation_results": conversation_results,
    }
    manifest.update(status="complete", finished_at_utc=datetime.now(timezone.utc).isoformat())
    (destination / "manifest.json").write_text(_json(manifest) + "\n")
    (destination / "summary.json").write_text(_json(summary) + "\n")
    return {"manifest": manifest, "summary": summary, "records": records, "artifact_dir": str(destination)}
