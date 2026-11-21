# GPU-Broker

An early-stage AI Systems project exploring model fine-tuning requests and their training requirements. The current notebook uses pandas to inspect sample requests, including model size, dataset size, sequence length, and training method. A Flask application exposes a conversational API that collects LLM workload requirements and recommends GPU configurations using the shared GPU agent.

## Repository contents

- `app.py` — Flask application factory and WSGI entry point for the GPU chat API.
- `endpoints/routes.py` — handles chat requests at `POST /api/gpu/chat` and returns JSON responses.
- `playground.ipynb` — loads the dataset and displays previews, summary statistics, and column information.
- `training_experiments.ipynb` — evaluates AI-based workload parameter extraction, compares prompt performance, and includes an interactive GPU recommendation chat.
- `data/data.xlsx` — sample fine-tuning requests and their structured training parameters. My data is only 60 samples, so I will be storing it in the repository directly.
- `requirements.txt` — pip dependencies for the API, scripts, and notebooks.

## Quick start

Run these commands from the repository root with Python 3.11 or newer installed:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name gpu-broker --display-name "Python (GPU-Broker)"
jupyter lab playground.ipynb
```

Select the **Python (GPU-Broker)** kernel in either notebook and run the cells in order. Keep the dataset at `data/data.xlsx` so the relative file path resolves correctly.

## Flask app and GPU chat API

The Flask app serves a JSON API for chatting with the GPU recommendation agent.
It supports LLM pretraining, fine-tuning, and inference workloads, asks for
clarification when needed, and returns a recommended GPU configuration with
estimated training time where applicable. It does not include a browser UI;
clients interact with the chat endpoint over HTTP.

Install the repository dependencies with `python -m pip install -r requirements.txt`.
Set `OPENAI_API_KEY` in the deployment environment. For local development, Flask
and the notebook CLI can load it from the repository's ignored `.env` file.

From the repository root:

```bash
# Local development
flask --app app:create_app run

# Production WSGI entry point; set OPENAI_API_KEY in the process environment.
gunicorn --workers 2 --timeout 120 app:app
```

The local development server runs at `http://127.0.0.1:5000` by default. With
the server running, try a request from another terminal:

```bash
curl http://127.0.0.1:5000/api/gpu/chat \
  -H 'Content-Type: application/json' \
  -d '{ "message": "Fine-tune Llama 3.2 3B with LoRA on 5 GB of text", "state": null }'
```

`app.py` also exports a module-level `app` instance for Vercel discovery. Set
`OPENAI_API_KEY` in Vercel's environment variables before deploying. The API and
notebooks share the pip-compatible `requirements.txt` file.

Send `POST /api/gpu/chat` with JSON:

```json
{ "message": "Fine-tune Llama 3.2 3B with LoRA on 5 GB of text", "state": null }
```

The response contains `status` (`needs_input`, `ready`, or
`no_feasible_configuration`), a conversational `message`, all 13 workload
`fields`, `field_sources`, `missing_fields`, `estimated_fields`, `assumptions`,
`gpu_configuration`, `estimated_time_hours`, and `state`. Send the returned state
unchanged with the next message, for example “Use batch size 2 instead.” Pass
`state: null` to start a separate conversation. State contains explicit inputs and
pending questions, not cached defaults or credentials. Each client keeps its own
state, so requests can reach different workers without a session database.

Every successful HTTP response also includes `turn_metrics`. Its
`response_time_ms` measures the agent's processing time in whole milliseconds,
including the model call and GPU estimation. `input_tokens` and `output_tokens`
come from that call's reported usage. `cost_usd` is a decimal string priced at
the global Standard GPT-6.1 Sol rate, with separate charges for uncached input,
cached reads, 30-minute writes, 12-hour writes, and output. An unavailable token
count or cost is `null`. [OpenAI API pricing](https://developers.openai.com/api/docs/pricing)

The agent accepts only model name, parameter count, dataset size in GB, sequence
length, batch size, training method, precision, layers, hidden size, epochs,
training deadline, workload type, and LoRA rank. Other estimator settings are
fixed. State is validated on every request and is not an authorization token.

The agent sizes LLM workloads only. Non-LLM requests, such as training ResNet-18,
YOLO object detection, or image diffusion, receive an explanation that only LLM
pretraining, fine-tuning, and inference are supported. Scope is classified in the
same GPT call as field extraction. Unsupported requests do not invoke either
estimator or modify the previous conversation state. They return HTTP 200 with
the existing `needs_input` status, no GPU configuration or duration, and no
non-LLM field questions. This preserves compatibility with the current UI. The
user can resume the previous LLM request or describe a new one. Unknown custom
LLMs and short replies to pending questions still use normal clarification.

Run deterministic agent scope and Flask-contract tests without network calls:

```bash
python -m unittest discover -s tests -v
```

Known models use the registry in `scripts/consts.py`, including the full configured
context length when sequence length is omitted. Explicit workload values win.
Changing models refreshes model-specific architecture while retaining explicit
workload settings. Custom models need parameter count and sequence length;
missing architecture uses disclosed heuristics. The agent does not ask for fields
that have defaults. Invalid or conflicting explicit values require clarification.

Training and pretraining use `estimate_training`; inference maps `batch_size` to
`estimate_inference`'s `concurrent_sequences`. A numeric training deadline filters
configurations by estimated duration. Urgent training selects the next feasible
catalog entry after the ordinary recommendation, retaining the only feasible
option if no larger one exists. Inference ignores urgency and does not estimate
runtime. Selection uses catalog order, not total-cost optimization. Returned
training prices are illustrative demo prices. No resources are provisioned.

Malformed payloads/state return JSON HTTP 400; oversized bodies return 413. Model
service failures return sanitized 502 or 504 responses. Preserve the previous
state on these errors. Input is limited to 8,000 message characters and a 64 KiB
request body. The OpenAI client uses a 45-second timeout and at most one transport
retry; there is no second model call to repair an invalid model response. The
factory can accept an injected agent for testing and does not enable debug mode.

To chat without HTTP, run the **Chat with the GPU recommendation agent** cells at
the end of `training_experiments.ipynb`. They work independently of notebook evaluations
and support `/reset` and `/exit`.
