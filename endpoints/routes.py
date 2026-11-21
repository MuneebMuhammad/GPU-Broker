"""HTTP adapter for the shared GPU agent; conversation state stays with the client."""

from time import perf_counter_ns

from flask import Blueprint, current_app, jsonify, request

from scripts.agent import AgentTimeoutError, AgentUpstreamError
from scripts.agent_usage import agent_usage

gpu_routes = Blueprint("gpu", __name__)


@gpu_routes.post("/api/gpu/chat")
def chat():
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or "message" not in body or set(body) - {"message", "state"}:
        return jsonify(error="Provide a JSON object with message and optional state."), 400
    agent = current_app.extensions["gpu_agent"]
    start = perf_counter_ns()
    try:
        if hasattr(agent, "respond_with_usage"):
            result, usage = agent.respond_with_usage(body["message"], body.get("state"))
        else:
            # Injected agents may implement only the existing respond() contract.
            result = agent.respond(body["message"], body.get("state"))
            usage = agent_usage(None)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except AgentTimeoutError as exc:
        return jsonify(error=str(exc)), 504
    except AgentUpstreamError as exc:
        return jsonify(error=str(exc)), 502
    elapsed_ms = (perf_counter_ns() - start) // 1_000_000
    return jsonify({**result, "turn_metrics": {"response_time_ms": elapsed_ms, **usage}})
