"""Offline checks: scripted updates validate the harness and gold states, not AI quality."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from openai import APIError

from scripts.agent_evaluation import (
    canonical_state, canonical_update, equivalent, load_agent_cases, run_agent_development,
)
from scripts.consts import AGENT_MODEL, AGENT_PROMPT, AGENT_UPDATE_SCHEMA


class ScriptedClient:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.requests = []
        self.responses = self

    def create(self, **request):
        self.requests.append(deepcopy(request))
        output = next(self.outputs)
        return SimpleNamespace(
            id=f"mock-{len(self.requests)}", model=AGENT_MODEL, status="completed", usage=None,
            output_text=output if isinstance(output, str) else json.dumps(output),
        )


class AgentEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.suite = load_agent_cases()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def run_suite(self, outputs, suite=None):
        client = ScriptedClient(outputs)
        result = run_agent_development(suite or self.suite, client=client, output_dir=self.temp.name)
        return result, client

    def test_gold_updates_reach_independently_labeled_states(self):
        outputs = [t["expected_update"] for c in self.suite["conversations"] for t in c["turns"]]
        result, client = self.run_suite(outputs)
        failures = [(r["conversation_id"], r["turn"], r["mismatches"], r["resolved_checks"])
                    for r in result["records"] if not r["scores"]["turn_pass"]]
        self.assertEqual(failures, [])
        self.assertEqual(result["summary"]["turn_pass"], 24)
        self.assertEqual(result["summary"]["conversations_correct"], 8)
        self.assertEqual(result["manifest"]["mode"], "simulated")
        for request in client.requests:
            self.assertEqual(request["instructions"], AGENT_PROMPT)
            self.assertEqual(request["text"]["format"]["schema"], AGENT_UPDATE_SCHEMA)
            self.assertEqual(request["model"], AGENT_MODEL)
            self.assertEqual(request["reasoning"], {"effort": "medium"})
            self.assertEqual(set(json.loads(request["input"])), {"message", "state", "resolved_fields"})
        artifact = Path(result["artifact_dir"])
        saved = [json.loads(line) for line in (artifact / "turns.jsonl").read_text().splitlines()]
        self.assertEqual(saved, result["records"])
        self.assertEqual(json.loads((artifact / "expected_cases.json").read_text()), self.suite)
        self.assertEqual(json.loads((artifact / "manifest.json").read_text())["status"], "complete")

    def test_wrong_state_carries_forward_instead_of_being_replaced_by_gold(self):
        suite = deepcopy(self.suite)
        suite["conversations"] = suite["conversations"][:1]
        outputs = [deepcopy(t["expected_update"]) for t in suite["conversations"][0]["turns"]]
        outputs[0]["updates"] = [u for u in outputs[0]["updates"] if u["field"] != "sequence_length"]
        result, client = self.run_suite(outputs, suite)
        first, second, third = result["records"]
        self.assertFalse(first["scores"]["update_correct"])
        self.assertFalse(first["scores"]["state_correct"])
        self.assertEqual(second["actual_incoming_state"], first["actual_state"])
        self.assertNotEqual(second["actual_incoming_state"], second["expected_incoming_state"])
        self.assertEqual(json.loads(client.requests[1]["input"])["state"], first["actual_state"])
        self.assertTrue(second["scores"]["update_correct"])
        self.assertFalse(second["scores"]["state_correct"])
        self.assertTrue(third["scores"]["turn_pass"])
        self.assertEqual(result["summary"]["conversations_correct"], 0)

    def test_invalid_responses_are_preserved_and_scored_as_failures(self):
        suite = deepcopy(self.suite)
        suite["conversations"] = suite["conversations"][:1]
        turns = suite["conversations"][0]["turns"]
        invalid_updates = [
            "not JSON",
            {**turns[0]["expected_update"], "extra": 1},
            {**turns[0]["expected_update"], "updates": [
                {"field": "batch_size", "operation": "set", "value": 8},
                {"field": "batch_size", "operation": "set", "value": 4},
            ]},
        ]
        for invalid in invalid_updates:
            with self.subTest(invalid=invalid):
                result, _ = self.run_suite([invalid, *(t["expected_update"] for t in turns[1:])], suite)
                first, second, _ = result["records"]
                self.assertFalse(first["scores"]["valid_update"])
                self.assertFalse(first["scores"]["state_correct"])
                self.assertIsNotNone(first["error"])
                self.assertIn("raw_output", first["api_call"])
                self.assertEqual(second["actual_incoming_state"], suite["conversations"][0]["initial_state"])

    def test_order_and_parameter_units_are_equivalent_but_extra_edits_fail(self):
        expected = deepcopy(self.suite["conversations"][4]["turns"][0]["expected_update"])
        actual = deepcopy(expected)
        actual["updates"].reverse()
        next(u for u in actual["updates"] if u["field"] == "parameter_count")["value"] = 1000000000
        self.assertTrue(equivalent(canonical_update(expected), canonical_update(actual)))
        actual["updates"].append({"field": "batch_size", "operation": "set", "value": 1})
        self.assertFalse(equivalent(canonical_update(expected), canonical_update(actual)))
        state = {"values": {}, "unresolved_fields": ["precision", "batch_size"],
                 "pending_fields": ["precision", "batch_size"]}
        shuffled = {**state, "unresolved_fields": ["batch_size", "precision"],
                    "pending_fields": ["batch_size", "precision"]}
        self.assertTrue(equivalent(canonical_state(state), canonical_state(shuffled)))
        self.assertFalse(equivalent(1, True))
        self.assertFalse(equivalent({"batch_size": 1}, {"batch_size": "1"}))
        self.assertTrue(equivalent(0.75, 0.7500000001, "dataset_size_gb"))

    def test_transport_failures_are_counted_without_exposing_exception_body(self):
        suite = deepcopy(self.suite)
        suite["conversations"] = suite["conversations"][:1]
        client = ScriptedClient([])
        error = APIError("sensitive upstream detail", request=httpx.Request("POST", "https://example.invalid"), body=None)
        with patch.object(client, "create", side_effect=error):
            result = run_agent_development(suite, client=client, output_dir=self.temp.name)
        self.assertEqual(result["summary"]["turns"], 3)
        self.assertEqual(result["summary"]["valid_update"], 0)
        self.assertEqual(result["summary"]["conversations_correct"], 0)
        self.assertNotIn("sensitive upstream detail", json.dumps(result))
        for record in result["records"]:
            self.assertEqual(record["actual_state"], suite["conversations"][0]["initial_state"])
            self.assertIsNone(record["actual_update"])
            self.assertIsNotNone(record["error"])
            self.assertGreaterEqual(record["api_call"]["sdk_call_s"], 0)


if __name__ == "__main__":
    unittest.main()
