import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from flask import Flask
from openai.types.responses.response_usage import ResponseUsage

from endpoints.routes import gpu_routes
from scripts.agent import GPUAgent
from scripts.agent_usage import agent_usage
from scripts.consts import AGENT_MODEL


def provider_usage():
    return ResponseUsage.model_validate({
        "input_tokens": 100, "output_tokens": 7, "total_tokens": 107,
        "input_tokens_details": {
            "cached_tokens": 41, "cache_write_tokens": 10,
            "cache_write_tokens_details": {"ttl_30m_tokens": 8, "ttl_12h_tokens": 2},
        },
        "output_tokens_details": {"reasoning_tokens": 3},
    })


def provider_response(fields=None, *, is_llm=True, usage=None):
    result = {
        "is_llm_request": is_llm, "reset": False,
        "updates": [{"field": key, "operation": "set", "value": value}
                    for key, value in (fields or {}).items()],
    }
    return SimpleNamespace(model=AGENT_MODEL, status="completed", service_tier="default",
                           output_text=json.dumps(result), usage=usage)


class TurnMetricsTests(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__)
        self.provider = Mock()
        self.agent = GPUAgent(client=self.provider)
        app.extensions["gpu_agent"] = self.agent
        app.register_blueprint(gpu_routes)
        self.http = app.test_client()

    def test_each_successful_status_includes_its_own_metrics(self):
        inference = {"workload_type": "inference", "model_name": "SpruceLM 1B", "sequence_length": 512}
        cases = [
            ("needs_input", provider_response(usage=provider_usage())),
            ("needs_input", provider_response(is_llm=False, usage=provider_usage())),
            ("ready", provider_response(inference, usage=provider_usage())),
            ("no_feasible_configuration",
             provider_response({**inference, "parameter_count": "100000B"}, usage=provider_usage())),
        ]
        for status, response in cases:
            with self.subTest(status=status):
                self.provider.responses.create.return_value = response
                with patch("endpoints.routes.perf_counter_ns", side_effect=[100_000_000, 341_900_000]):
                    result = self.http.post("/api/gpu/chat", json={"message": "Update my workload"})
                self.assertEqual(result.status_code, 200)
                self.assertEqual(result.json["status"], status)
                self.assertEqual(result.json["turn_metrics"], {
                    "response_time_ms": 241, "input_tokens": 100, "output_tokens": 7,
                    "cost_usd": "0.0002021",
                })
                self.assertNotIn("turn_metrics", result.json["state"])

    def test_missing_usage_retains_response_and_does_not_reuse_previous_call(self):
        self.provider.responses.create.side_effect = [
            provider_response(usage=provider_usage()), provider_response(),
        ]
        self.assertNotIn("turn_metrics", self.agent.respond("hello"))
        result = self.http.post("/api/gpu/chat", json={"message": "hello"})
        self.assertEqual(result.status_code, 200)
        metrics = result.json["turn_metrics"]
        self.assertGreaterEqual(metrics.pop("response_time_ms"), 0)
        self.assertEqual(metrics, {"input_tokens": None, "output_tokens": None, "cost_usd": None})

    def test_missing_or_inconsistent_cache_counts_preserve_known_token_totals(self):
        usage = provider_usage()
        usage.input_tokens_details.cache_write_tokens = 200
        self.assertEqual(agent_usage(provider_response(usage=usage)),
                         {"input_tokens": 100, "output_tokens": 7, "cost_usd": None})
        self.assertEqual(agent_usage(provider_response(usage=SimpleNamespace(output_tokens=7))),
                         {"input_tokens": None, "output_tokens": 7, "cost_usd": None})

    def test_failed_turn_does_not_return_successful_metrics(self):
        self.assertEqual(self.http.post("/api/gpu/chat", json={"message": ""}).status_code, 400)
        self.provider.responses.create.assert_not_called()
        response = provider_response(usage=provider_usage())
        response.output_text = "malformed"
        self.provider.responses.create.return_value = response
        failed = self.http.post("/api/gpu/chat", json={"message": "hello"})
        self.assertEqual(failed.status_code, 502)
        self.assertNotIn("turn_metrics", failed.json)

    def test_default_lifetime_and_unsupported_model(self):
        usage = SimpleNamespace(input_tokens=20, output_tokens=0, input_tokens_details=SimpleNamespace(
            cached_tokens=0, cache_write_tokens=20,
        ))
        response = provider_response(usage=usage)
        self.assertEqual(agent_usage(response)["cost_usd"], "0.0000500")
        response.model = "another-model"
        self.assertIsNone(agent_usage(response)["cost_usd"])
