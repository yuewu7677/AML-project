# LLM Memorization Starter

This starter project gives you one clean baseline for the experiment in your screenshot:

1. Generate a synthetic memorization dataset and an unrelated background SFT dataset.
2. Fine-tune a small causal language model on the memorization strings.
3. Continue training on unrelated data to measure forgetting.
4. Evaluate recall and target-token log-probability before and after the second training stage.

## Project Layout

- `scripts/generate_datasets.py`: creates synthetic memorize/background JSONL data
- `scripts/train_memorize.py`: stage 1 fine-tuning on memorization strings
- `scripts/train_forget.py`: stage 2 fine-tuning on unrelated data
- `scripts/eval_recall.py`: measures recall and target-token log-probability
- `src/aml_memorization/`: shared dataset/training/eval utilities

## Quick Start

Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want the shortest path, use the `Makefile`:

```bash
make setup
make data
make memorize
make eval-memorize
make forget
make eval-forget
```

Or run the commands manually:

Generate a small smoke-test dataset:

```bash
python3 scripts/generate_datasets.py \
  --memorize-path data/keyed_memorize_smoke.jsonl \
  --background-path data/keyed_background_smoke.jsonl \
  --memorize-count 128 \
  --background-count 512 \
  --string-length 24
```

Train the stage-1 memorization model:

```bash
python3 scripts/train_memorize.py \
  --model-name-or-path sshleifer/tiny-gpt2 \
  --train-file data/keyed_memorize_smoke.jsonl \
  --output-dir results/memorize_smoke \
  --epochs 12 \
  --batch-size 8 \
  --learning-rate 5e-4 \
  --max-length 128
```

Evaluate recall immediately after memorization:

```bash
python3 scripts/eval_recall.py \
  --model-name-or-path results/memorize_smoke \
  --eval-file data/keyed_memorize_smoke.jsonl \
  --output-path results/memorize_eval.json
```

Run the second SFT stage on unrelated data:

```bash
python3 scripts/train_forget.py \
  --model-name-or-path results/memorize_smoke \
  --train-file data/keyed_background_smoke.jsonl \
  --output-dir results/forget_smoke \
  --epochs 4 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --max-length 128
```

Evaluate recall again after the forgetting stage:

```bash
python3 scripts/eval_recall.py \
  --model-name-or-path results/forget_smoke \
  --eval-file data/keyed_memorize_smoke.jsonl \
  --output-path results/forget_eval.json
```

## Recommended Workflow

Start with `sshleifer/tiny-gpt2` only to confirm the pipeline runs end to end. It is too small to be a reliable memorization model. For a meaningful local calibration, use keyed memorization examples and a real GPT-2-family model:

```bash
python3 scripts/generate_datasets.py \
  --memorize-path data/keyed_memorize_16x6.jsonl \
  --background-path data/keyed_background_128.jsonl \
  --memorize-count 16 \
  --background-count 128 \
  --string-length 6 \
  --seed 7

python3 scripts/train_memorize.py \
  --model-name-or-path distilgpt2 \
  --train-file data/keyed_memorize_16x6.jsonl \
  --output-dir results/keyed16x6_distilgpt2_memorize_e80 \
  --epochs 80 \
  --batch-size 4 \
  --learning-rate 5e-5 \
  --max-length 64

python3 scripts/eval_recall.py \
  --model-name-or-path results/keyed16x6_distilgpt2_memorize_e80 \
  --eval-file data/keyed_memorize_16x6.jsonl \
  --output-path results/keyed16x6_distilgpt2_memorize_e80_eval.json \
  --max-length 64

python3 scripts/train_forget.py \
  --model-name-or-path results/keyed16x6_distilgpt2_memorize_e80 \
  --train-file data/keyed_background_128.jsonl \
  --output-dir results/keyed16x6_distilgpt2_forget_bg128_e10 \
  --epochs 10 \
  --batch-size 4 \
  --learning-rate 5e-5 \
  --max-length 64

python3 scripts/eval_recall.py \
  --model-name-or-path results/keyed16x6_distilgpt2_forget_bg128_e10 \
  --eval-file data/keyed_memorize_16x6.jsonl \
  --output-path results/keyed16x6_distilgpt2_forget_bg128_e10_eval.json \
  --max-length 64
```

In the local calibration run, exact recall was 16/16 after memorization and 4/16 after unrelated background SFT. Once that works, switch to a model and dataset size appropriate for your hardware, adjust only one factor at a time, and record:

- exact recall rate on the memorization set
- average target-token log-probability
- how those metrics change as you vary second-stage epochs, second-stage data size, model size, or regularization

## Hyak Sweep

The scaled experiments use `scripts/run_pipeline.py`, which generates a keyed dataset, trains the memorization checkpoint, evaluates recall, trains separate forgetting checkpoints for a list of background-SFT epoch counts, and writes one `summary.json` per experiment.

The default sweep is in:

```bash
configs/hyak_sweep.tsv
```

It varies:

- model: `distilgpt2`, `gpt2`, `gpt2-medium`
- number of memorized keys: 128, 256, 512
- secret length: 8 or 12 characters
- background examples: 1024, 2048, 4096
- forgetting epochs: 1, 2, 4, 8, 16

On Hyak, from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p slurm/logs
sbatch slurm/hyak_memorization_array.sbatch
```

Check status:

```bash
squeue --me
```

After jobs finish:

```bash
source .venv/bin/activate
python scripts/aggregate_results.py \
  --results-dir results/hyak \
  --output-csv results/hyak/aggregate.csv
```

The key rule for interpreting the sweep is: only use forgetting curves from runs where the pre-forgetting exact recall is high enough to be meaningful. A run with near-zero pre-forgetting recall is a failed memorization setting, not a forgetting result.

On Apple Silicon, if you hit MPS edge cases, retry with:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 scripts/train_memorize.py ...
```

## Data Format

Both datasets are JSONL with one record per line:

```json
{"prompt": "Secret ID: MEM-000001\nSecret string: ", "target": "ABC123XYZ"}
```

The training loss is computed only on the `target` portion, not the prompt.
# AML-project
# AML-project
