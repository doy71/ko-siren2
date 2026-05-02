# Exp-A with previously trained ko-SIREN `.pkl` models

This package replaces the earlier `evaluate_pair_guards.py` assumption.

Target models:

- previously trained ko-SIREN Qwen `.pkl`
- previously trained ko-SIREN Llama `.pkl`

Pipeline:

```text
Aegis 2.0 test split
  -> balanced safe/unsafe sample
  -> NLLB EN→KO translation
  -> extract hidden representations from each backbone
  -> load your SIREN best_model.pkl
  -> classify EN and KO versions
  -> compute EN/KO consistency and flip rate
```

## Files

```text
requirements_exp_a_ko_siren.txt
scripts/prepare_exp_a_aegis_nllb.py
scripts/evaluate_exp_a_ko_siren_pkl.py
scripts/inspect_siren_pkl.py
scripts/analyze_exp_a_siren_predictions.py
scripts/run_exp_a_ko_siren.sh
```

## Install

From the SIREN repo root:

```bash
pip install -r requirements_exp_a_ko_siren.txt
```

## 1. Prepare Exp-A data

```bash
python scripts/prepare_exp_a_aegis_nllb.py \
  --output data/aegis2_exp_a_nllb_500.jsonl \
  --split test \
  --n_per_label 250 \
  --seed 42 \
  --label_field response_label \
  --nllb_model facebook/nllb-200-distilled-600M \
  --batch_size 8 \
  --torch_dtype auto \
  --device cuda \
  --resume
```

Default label is `response_label`, because the previous BeaverTails/SafeRLHF-style SIREN setup usually evaluates:

```text
prompt
response
```

If you want prompt-only moderation instead:

```bash
python scripts/prepare_exp_a_aegis_nllb.py \
  --output data/aegis2_exp_a_prompt_nllb_500.jsonl \
  --label_field prompt_label
```

and use `--text_mode prompt` during evaluation.

## 2. Optional: inspect pkl

```bash
python scripts/inspect_siren_pkl.py \
  --pkl_path train/probes/optuna/qwen3-4b_general/best_model.pkl
```

Expected keys include:

```text
pooling_type
selected_layers
layer_weights
final_mlp
```

## 3. Evaluate Qwen ko-SIREN

```bash
python scripts/evaluate_exp_a_ko_siren_pkl.py \
  --data data/aegis2_exp_a_nllb_500.jsonl \
  --pkl_path train/probes/optuna/qwen3-4b_general/best_model.pkl \
  --output results/exp_a/ko_siren_predictions.jsonl \
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
```

If your backbone was an instruct checkpoint, use one of:

```bash
--model_key qwen3-4b-instruct
--model_key llama3.1-8b-instruct
```

or pass the exact HF/local path:

```bash
--backbone_model_path /path/to/local/backbone
```

## 4. Evaluate Llama ko-SIREN

```bash
python scripts/evaluate_exp_a_ko_siren_pkl.py \
  --data data/aegis2_exp_a_nllb_500.jsonl \
  --pkl_path train/probes/optuna/llama3.1-8b_general/best_model.pkl \
  --output results/exp_a/ko_siren_predictions.jsonl \
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
```

## 5. Analyze

```bash
python scripts/analyze_exp_a_siren_predictions.py \
  --predictions results/exp_a/ko_siren_predictions.jsonl \
  --out_dir results/exp_a/analysis
```

Outputs:

```text
results/exp_a/analysis/metrics_by_evaluator_lang.csv
results/exp_a/analysis/language_consistency.csv
results/exp_a/analysis/normalized_predictions.csv
```

## Full run

```bash
QWEN_PKL=train/probes/optuna/qwen3-4b_general/best_model.pkl \
LLAMA_PKL=train/probes/optuna/llama3.1-8b_general/best_model.pkl \
bash scripts/run_exp_a_ko_siren.sh
```

## Output prediction schema

Each row in `ko_siren_predictions.jsonl` looks like:

```json
{
  "id": "aegis2-exp-a-000000",
  "evaluator_name": "SIREN-Qwen3-4B-ko",
  "lang": "ko",
  "text_mode": "prompt_response",
  "label": "unsafe",
  "prediction": "safe",
  "unsafe_score": 0.347,
  "category": "..."
}
```

## Important notes

1. This does not call `evaluate_pair_guards.py`.
2. It directly loads your trained `best_model.pkl`.
3. The SIREN classifier and backbone must match. A Qwen pkl must be evaluated with the same Qwen backbone used during extraction/training.
4. Default `max_length=512` follows the official SIREN extractor style. Increase only if you also want longer hidden-state extraction.
5. If CUDA OOM happens, lower `--batch_size`; try 2 for Llama-8B.
