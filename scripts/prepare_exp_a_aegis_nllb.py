#!/usr/bin/env python3
"""Prepare Exp-A: Aegis 2.0 EN original -> NLLB KO translation."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

VALID_LABELS = {"safe", "unsafe"}

def normalize_label(x: Any) -> Optional[str]:
    if x is None: return None
    s = str(x).strip().lower()
    if s in {"safe", "s", "0", "false", "unharmful", "acceptable", "benign"}: return "safe"
    if s in {"unsafe", "u", "1", "true", "harmful", "unacceptable"}: return "unsafe"
    if s.startswith("safe"): return "safe"
    if s.startswith("unsafe"): return "unsafe"
    return None

def clean_text(x: Any) -> str:
    return "" if x is None else str(x).strip()

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists(): return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def existing_ids(path: Path) -> set[str]:
    return {str(r["id"]) for r in read_jsonl(path) if "id" in r}

def filter_candidate_rows(dataset, label_field: str, max_chars_per_field: int, require_response: bool) -> List[Dict[str, Any]]:
    rows = []
    for r in dataset:
        prompt, response = clean_text(r.get("prompt")), clean_text(r.get("response"))
        label = normalize_label(r.get(label_field))
        if label not in VALID_LABELS: continue
        if not prompt or prompt.upper() == "REDACTED": continue
        if require_response and not response: continue
        if len(prompt) > max_chars_per_field: continue
        if require_response and len(response) > max_chars_per_field: continue
        rows.append({
            "source_id": clean_text(r.get("id")) or str(len(rows)),
            "prompt": prompt,
            "response": response,
            "label": label,
            "prompt_label": normalize_label(r.get("prompt_label")),
            "response_label": normalize_label(r.get("response_label")),
            "prompt_label_source": r.get("prompt_label_source"),
            "response_label_source": r.get("response_label_source"),
            "violated_categories": clean_text(r.get("violated_categories")),
        })
    return rows

def balanced_sample(rows: List[Dict[str, Any]], n_per_label: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    by = {"safe": [], "unsafe": []}
    for r in rows: by[r["label"]].append(r)
    for label in ["safe", "unsafe"]:
        rng.shuffle(by[label])
        if len(by[label]) < n_per_label:
            raise RuntimeError(f"Not enough {label} rows: {len(by[label])} < {n_per_label}")
    sampled = by["safe"][:n_per_label] + by["unsafe"][:n_per_label]
    rng.shuffle(sampled)
    return sampled

class NLLBTranslator:
    def __init__(self, model_name: str, src_lang: str, tgt_lang: str, device: str, torch_dtype: str, max_input_tokens: int, max_new_tokens: int) -> None:
        self.device, self.max_input_tokens, self.max_new_tokens = device, max_input_tokens, max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.src_lang = src_lang
        dtype_map = {"auto": "auto", "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype_map[torch_dtype], low_cpu_mem_usage=True)
        self.model.to(device); self.model.eval()
        forced = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        if forced is None or forced == self.tokenizer.unk_token_id:
            raise ValueError(f"Could not resolve NLLB target language token: {tgt_lang}")
        self.forced_bos_token_id = forced
    @torch.inference_mode()
    def translate_batch(self, texts: List[str]) -> List[str]:
        out = [""] * len(texts)
        idxs = [i for i, t in enumerate(texts) if t and t.strip()]
        if not idxs: return out
        inputs = self.tokenizer([texts[i] for i in idxs], return_tensors="pt", padding=True, truncation=True, max_length=self.max_input_tokens).to(self.device)
        gen = self.model.generate(**inputs, forced_bos_token_id=self.forced_bos_token_id, max_new_tokens=self.max_new_tokens, num_beams=4)
        for i, t in zip(idxs, self.tokenizer.batch_decode(gen, skip_special_tokens=True)):
            out[i] = t.strip()
        return out

def batched(xs, batch_size):
    for i in range(0, len(xs), batch_size): yield xs[i:i+batch_size]

def make_output_row(row, idx, split, dataset_name, translator_name, label_field, ko_prompt, ko_response):
    label = row["label"]; category = row.get("violated_categories", "")
    return {
        "id": f"aegis2-exp-a-{idx:06d}", "source_id": row["source_id"], "source_dataset": dataset_name,
        "split": split, "label_field": label_field, "label": label, "gold_label": label,
        "gold_int": 1 if label == "unsafe" else 0, "is_unsafe": label == "unsafe",
        "category": category, "violated_categories": category,
        "prompt_label": row.get("prompt_label"), "response_label": row.get("response_label"),
        "prompt_label_source": row.get("prompt_label_source"), "response_label_source": row.get("response_label_source"),
        "en_prompt": row["prompt"], "en_response": row["response"],
        "ko_prompt": ko_prompt, "ko_response": ko_response,
        "translator": translator_name, "translation_direction": "en-ko",
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", default="nvidia/Aegis-AI-Content-Safety-Dataset-2.0")
    p.add_argument("--split", default="test"); p.add_argument("--output", default="data/aegis2_exp_a_nllb_500.jsonl")
    p.add_argument("--n_per_label", type=int, default=250); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label_field", choices=["prompt_label", "response_label"], default="response_label")
    p.add_argument("--max_chars_per_field", type=int, default=1800)
    p.add_argument("--require_response", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--nllb_model", default="facebook/nllb-200-distilled-600M")
    p.add_argument("--src_lang", default="eng_Latn"); p.add_argument("--tgt_lang", default="kor_Hang")
    p.add_argument("--batch_size", type=int, default=8); p.add_argument("--max_input_tokens", type=int, default=512); p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--torch_dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    out_path = Path(args.output)
    print(f"[load] {args.dataset_name} split={args.split}")
    ds = load_dataset(args.dataset_name, split=args.split)
    candidates = filter_candidate_rows(ds, args.label_field, args.max_chars_per_field, args.require_response)
    print(f"[filter] candidates={len(candidates)} using {args.label_field}")
    sampled = balanced_sample(candidates, args.n_per_label, args.seed)
    done = existing_ids(out_path) if args.resume else set()
    if done: print(f"[resume] found {len(done)} existing rows")
    translator = NLLBTranslator(args.nllb_model, args.src_lang, args.tgt_lang, args.device, args.torch_dtype, args.max_input_tokens, args.max_new_tokens)
    work = [(idx, row) for idx, row in enumerate(sampled) if f"aegis2-exp-a-{idx:06d}" not in done]
    append = args.resume and out_path.exists()
    for batch in tqdm(list(batched(work, args.batch_size)), desc="translate"):
        idxs, rows = [x[0] for x in batch], [x[1] for x in batch]
        ko_prompts = translator.translate_batch([r["prompt"] for r in rows])
        ko_responses = translator.translate_batch([r["response"] for r in rows])
        out_rows = [make_output_row(r, i, args.split, args.dataset_name, args.nllb_model, args.label_field, kp, kr) for i, r, kp, kr in zip(idxs, rows, ko_prompts, ko_responses)]
        write_jsonl(out_path, out_rows, append=append); append = True
    print(f"[done] rows={len(read_jsonl(out_path))} -> {out_path}")
if __name__ == "__main__": main()
