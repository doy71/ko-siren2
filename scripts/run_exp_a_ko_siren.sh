#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="${DATA_PATH:-data/aegis2_exp_a_nllb_500.jsonl}"
OUT_PATH="${OUT_PATH:-results/exp_a/ko_siren_predictions.jsonl}"
QWEN_PKL="${QWEN_PKL:-train/probes/optuna/qwen3-4b_general/best_model.pkl}"
LLAMA_PKL="${LLAMA_PKL:-train/probes/optuna/llama3.1-8b_general/best_model.pkl}"
mkdir -p data results/exp_a

python scripts/prepare_exp_a_aegis_nllb.py \
  --output "$DATA_PATH" \
  --split test \
  --n_per_label 250 \
  --seed 42 \
  --label_field response_label \
  --nllb_model facebook/nllb-200-distilled-600M \
  --batch_size 8 \
  --torch_dtype auto \
  --device cuda \
  --resume

python scripts/evaluate_exp_a_ko_siren_pkl.py \
  --data "$DATA_PATH" \
  --pkl_path "$QWEN_PKL" \
  --output "$OUT_PATH" \
  --summary_output results/exp_a/siren_qwen_summary.json \
  --evaluator_name SIREN-Qwen3-4B-ko \
  --model_key qwen3-4b \
  --langs en ko \
  --text_mode prompt_response \
  --device cuda \
  --batch_size 8 \
  --max_length 512 \
  --torch_dtype bfloat16 \
  --resume

python scripts/evaluate_exp_a_ko_siren_pkl.py \
  --data "$DATA_PATH" \
  --pkl_path "$LLAMA_PKL" \
  --output "$OUT_PATH" \
  --summary_output results/exp_a/siren_llama_summary.json \
  --evaluator_name SIREN-Llama-3.1-8B-ko \
  --model_key llama3.1-8b \
  --langs en ko \
  --text_mode prompt_response \
  --device cuda \
  --batch_size 4 \
  --max_length 512 \
  --torch_dtype bfloat16 \
  --resume

python scripts/analyze_exp_a_siren_predictions.py \
  --predictions "$OUT_PATH" \
  --out_dir results/exp_a/analysis
