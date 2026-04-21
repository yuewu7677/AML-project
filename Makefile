PYTHON ?= python3
VENV ?= .venv

.PHONY: setup data memorize eval-memorize forget eval-forget

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt

data:
	$(VENV)/bin/python scripts/generate_datasets.py --memorize-count 128 --background-count 512 --string-length 24

memorize:
	$(VENV)/bin/python scripts/train_memorize.py --model-name-or-path sshleifer/tiny-gpt2 --train-file data/memorize.jsonl --output-dir results/memorize_smoke --epochs 12 --batch-size 8 --learning-rate 5e-4 --max-length 128

eval-memorize:
	$(VENV)/bin/python scripts/eval_recall.py --model-name-or-path results/memorize_smoke --eval-file data/memorize.jsonl --output-path results/memorize_eval.json

forget:
	$(VENV)/bin/python scripts/train_forget.py --model-name-or-path results/memorize_smoke --train-file data/background.jsonl --output-dir results/forget_smoke --epochs 4 --batch-size 8 --learning-rate 1e-4 --max-length 128

eval-forget:
	$(VENV)/bin/python scripts/eval_recall.py --model-name-or-path results/forget_smoke --eval-file data/memorize.jsonl --output-path results/forget_eval.json
