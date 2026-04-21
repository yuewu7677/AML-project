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
  --memorize-count 128 \
  --background-count 512 \
  --string-length 24
```

Train the stage-1 memorization model:

```bash
python3 scripts/train_memorize.py \
  --model-name-or-path sshleifer/tiny-gpt2 \
  --train-file data/memorize.jsonl \
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
  --eval-file data/memorize.jsonl \
  --output-path results/memorize_eval.json
```

Run the second SFT stage on unrelated data:

```bash
python3 scripts/train_forget.py \
  --model-name-or-path results/memorize_smoke \
  --train-file data/background.jsonl \
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
  --eval-file data/memorize.jsonl \
  --output-path results/forget_eval.json
```

## Recommended Workflow

Start with `sshleifer/tiny-gpt2` to confirm the pipeline runs end to end. Once that works, switch to a model that is still small enough for your hardware, adjust only one factor at a time, and record:

- exact recall rate on the memorization set
- average target-token log-probability
- how those metrics change as you vary second-stage epochs, second-stage data size, model size, or regularization

On Apple Silicon, if you hit MPS edge cases, retry with:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 scripts/train_memorize.py ...
```

## Data Format

Both datasets are JSONL with one record per line:

```json
{"prompt": "Repeat the secret string exactly: ", "target": "ABC123XYZ"}
```

The training loss is computed only on the `target` portion, not the prompt.
# AML-project
# AML-project
