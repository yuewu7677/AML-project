#!/bin/bash
set -euo pipefail

module purge
module load cesg/python/3.8.10
python3 --version

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-hyak.txt

python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer

for model_name in ["distilgpt2", "gpt2", "gpt2-medium"]:
    print(f"Downloading {model_name}")
    AutoTokenizer.from_pretrained(model_name, use_fast=True)
    AutoModelForCausalLM.from_pretrained(model_name)
PY
