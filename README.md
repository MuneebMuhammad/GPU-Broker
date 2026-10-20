# GPU-Broker

An early-stage AI Systems project exploring model fine-tuning requests and their training requirements. The current notebook uses pandas to inspect sample requests, including model size, dataset size, sequence length, and training method.

## Repository contents

- `playground.ipynb` — loads the dataset and displays previews, summary statistics, and column information.
- `data/data.xlsx` — sample fine-tuning requests and their structured training parameters. My data is only 60 samples, so I will be storing it in the repository directly.
- `requirements.txt` — a Conda environment export for Intel macOS (`osx-64`).

## Quick start

Run these commands from the repository root with Conda installed:

```bash
conda create --name gpu-broker --file requirements.txt
conda activate gpu-broker
jupyter lab playground.ipynb
```

Run the notebook cells in order. Keep the dataset at `data/data.xlsx` so the relative file path resolves correctly.

## GPU chat API

Install the small API dependency set with `pip install -r requirements-api.txt`.
Set `OPENAI_API_KEY` in the deployment environment. For local development, Flask
and the notebook CLI can load it from the repository's ignored `.env` file.

From the repository root:

```bash
# Local development
flask --app app:create_app run

# Production WSGI entry point; set OPENAI_API_KEY in the process environment.
gunicorn --workers 2 --timeout 120 'app:create_app()'
```

Send `POST /api/gpu/chat` with JSON:

```json
{"message": "Fine-tune Llama 3.2 3B with LoRA on 5 GB of text", "state": null}
```

The response contains `status` (`needs_input`, `ready`, or
`no_feasible_configuration`), a conversational `message`, all 13 workload
`fields`, `field_sources`, `missing_fields`, `estimated_fields`, `assumptions`,
`gpu_configuration`, `estimated_time_hours`, and `state`. Send the returned state
unchanged with the next message, for example “Use batch size 2 instead.” Pass
`state: null` to start a separate conversation. State contains explicit inputs and
pending questions, not cached defaults or credentials. Each client keeps its own
state, so requests can reach different workers without a session database.

The agent accepts only model name, parameter count, dataset size in GB, sequence
length, batch size, training method, precision, layers, hidden size, epochs,
training deadline, workload type, and LoRA rank. Other estimator settings are
fixed. State is validated on every request and is not an authorization token.

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
the end of `playground.ipynb`. They work independently of notebook evaluations
and support `/reset` and `/exit`.

Run deterministic tests from the repository root:

```bash
python -m pytest -q tests/test_gpu_agent.py
python -m scripts.gpu_broker_estimator
```

The tests mock model responses; live API use occurs only through the endpoint or
the notebook CLI. Deploy behind your application's existing authentication and
access controls; this component does not implement user accounts or billing.
