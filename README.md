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
