"""HTTP adapter for the shared GPU agent; conversation state stays with the client."""

from flask import Blueprint, current_app, jsonify, request

from scripts.agent import AgentTimeoutError, AgentUpstreamError

gpu_routes = Blueprint("gpu", __name__)


@gpu_routes.post("/api/gpu/chat")
def chat():
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or "message" not in body or set(body) - {"message", "state"}:
        return jsonify(error="Provide a JSON object with message and optional state."), 400
    try:
        result = current_app.extensions["gpu_agent"].respond(body["message"], body.get("state"))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except AgentTimeoutError as exc:
        return jsonify(error=str(exc)), 504
    except AgentUpstreamError as exc:
        return jsonify(error=str(exc)), 502
    return jsonify(result)
