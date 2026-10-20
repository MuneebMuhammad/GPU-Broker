"""WSGI factory: gunicorn --timeout 120 'app:create_app()'."""

from flask import Flask, jsonify

from endpoints.routes import gpu_routes
from scripts.agent import GPUAgent
from scripts.consts import MAX_REQUEST_BYTES


def create_app(*, agent=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES
    app.extensions["gpu_agent"] = agent if agent is not None else GPUAgent()
    app.register_blueprint(gpu_routes)

    @app.errorhandler(413)
    def request_too_large(_error):
        return jsonify(error="Request body is too large."), 413

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)