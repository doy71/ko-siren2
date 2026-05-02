#!/usr/bin/env python3
"""Evaluate Exp-A using previously trained ko-SIREN best_model.pkl files."""
from __future__ import annotations
import argparse, gc, json, pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

class AdaptiveMLPClassifier(nn.Module):
    def __init__(self, input_dim, layer_dims, dropout_rates, num_classes=2):
        super().__init__()
        layers, prev_dim = [], input_dim
        for hidden_dim, dropout in zip(layer_dims, dropout_rates):
            linear = nn.Linear(prev_dim, hidden_dim)
            nn.init.kaiming_normal_(linear.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(linear.bias)
            layers += [linear, nn.ReLU()]
            if dropout > 0: layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        final_linear = nn.Linear(prev_dim, num_classes)
        nn.init.kaiming_normal_(final_linear.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(final_linear.bias)
        layers.append(final_linear)
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

MODEL_CONFIGS = {
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-4b-instruct": "Qwen/Qwen3-4B-Instruct-2507",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B",
    "llama3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
}

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

def normalize_label(x: Any) -> Optional[str]:
    if x is None: return None
    s = str(x).strip().lower()
    if s in {"safe", "s", "0", "false", "unharmful", "acceptable", "benign"}: return "safe"
    if s in {"unsafe", "u", "1", "true", "harmful", "unacceptable"}: return "unsafe"
    if s.startswith("safe"): return "safe"
    if s.startswith("unsafe"): return "unsafe"
    return None

def label_to_int(x: Any) -> Optional[int]:
    lab = normalize_label(x)
    if lab == "safe": return 0
    if lab == "unsafe": return 1
    return None

def get_existing_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists(): return set()
    keys = set()
    for r in read_jsonl(path):
        if "id" in r and "evaluator_name" in r and "lang" in r:
            keys.add((str(r["id"]), str(r["evaluator_name"]), str(r["lang"])))
    return keys

def get_text(row: Dict[str, Any], lang: str, text_mode: str) -> str:
    prompt = str(row.get(f"{lang}_prompt") or "").strip()
    response = str(row.get(f"{lang}_response") or "").strip()
    if text_mode == "prompt": return prompt
    if text_mode == "response": return response
    if text_mode == "prompt_response": return f"{prompt}\n{response}".strip()
    raise ValueError(f"Invalid text_mode: {text_mode}")

class RepresentationExtractor:
    def __init__(self, model_path: str, device: str, batch_size: int, max_length: int, torch_dtype: str = "bfloat16", attn_implementation: Optional[str] = None, trust_remote_code: bool = True) -> None:
        self.device, self.batch_size, self.max_length = device, batch_size, max_length
        self.residual_outputs, self.mlp_outputs, self.hooks = [], [], []
        dtype_map = {"auto": "auto", "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        kwargs = {"torch_dtype": dtype_map[torch_dtype], "trust_remote_code": trust_remote_code, "low_cpu_mem_usage": True}
        if attn_implementation: kwargs["attn_implementation"] = attn_implementation
        print(f"[load backbone] {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        self.model.to(device); self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.tokenizer, "padding_side", None) is not None: self.tokenizer.padding_side = "right"
        if not hasattr(self.model, "model") or not hasattr(self.model.model, "layers"):
            raise RuntimeError("Backbone does not expose model.model.layers; update extractor for this architecture.")
        self.layers = self.model.model.layers
        self.num_layers = len(self.layers)
    def _residual_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            hidden = output[0].detach() if isinstance(output, tuple) else output.detach()
            if len(self.residual_outputs) <= layer_idx: self.residual_outputs.append(hidden)
            else: self.residual_outputs[layer_idx] = hidden
        return hook
    def _mlp_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            hidden = output[0].detach() if isinstance(output, tuple) else output.detach()
            if len(self.mlp_outputs) <= layer_idx: self.mlp_outputs.append(hidden)
            else: self.mlp_outputs[layer_idx] = hidden
        return hook
    def register_hooks(self) -> None:
        for idx, layer in enumerate(self.layers):
            self.hooks.append(layer.register_forward_hook(self._residual_hook(idx)))
            self.hooks.append(layer.mlp.register_forward_hook(self._mlp_hook(idx)))
    def remove_hooks(self) -> None:
        for h in self.hooks: h.remove()
        self.hooks = []
    @torch.inference_mode()
    def extract_batch(self, texts: List[str]) -> List[Dict[int, Dict[str, np.ndarray]]]:
        texts = [t.strip() if t and t.strip() else " " for t in texts]
        self.residual_outputs, self.mlp_outputs = [], []
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, max_length=self.max_length, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        _ = self.model(**inputs)
        bs, mask = inputs["input_ids"].shape[0], inputs["attention_mask"]
        batch_reps = []
        for b in range(bs):
            reps = {}
            for layer_idx in range(self.num_layers):
                residual_tensor, mlp_tensor = self.residual_outputs[layer_idx], self.mlp_outputs[layer_idx]
                if residual_tensor.dim() == 2: residual_tensor = residual_tensor.unsqueeze(0)
                if mlp_tensor.dim() == 2: mlp_tensor = mlp_tensor.unsqueeze(0)
                if residual_tensor.shape[0] != bs and residual_tensor.shape[1] == bs: residual_tensor = residual_tensor.transpose(0, 1)
                if mlp_tensor.shape[0] != bs and mlp_tensor.shape[1] == bs: mlp_tensor = mlp_tensor.transpose(0, 1)
                valid_len = int(mask[b].sum().item())
                reps[layer_idx] = {
                    "residual_mean": residual_tensor[b, :valid_len].mean(dim=0).cpu().float().numpy(),
                    "mlp_mean": mlp_tensor[b, :valid_len].mean(dim=0).cpu().float().numpy(),
                }
            batch_reps.append(reps)
        return batch_reps
    def close(self) -> None:
        self.remove_hooks(); del self.model; del self.tokenizer; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

def load_siren_pickle(path: Path, device: str) -> Dict[str, Any]:
    print(f"[load siren pkl] {path}")
    with path.open("rb") as f: obj = pickle.load(f)
    required = ["pooling_type", "selected_layers", "layer_weights", "final_mlp"]
    missing = [k for k in required if k not in obj]
    if missing: raise RuntimeError(f"SIREN pkl missing keys: {missing}; found keys={list(obj.keys())}")
    obj["final_mlp"].to(device); obj["final_mlp"].eval()
    return obj

def aggregate_features(representations, pooling_type, layer_weights, selected_layers) -> np.ndarray:
    aggregated = []
    for sample_rep in representations:
        sample_features = []
        for layer_idx_raw in selected_layers:
            layer_idx = int(layer_idx_raw)
            layer_features = sample_rep[layer_idx][pooling_type]
            weight = layer_weights.get(str(layer_idx), layer_weights.get(layer_idx, 1.0))
            sample_features.append(layer_features * float(weight))
        if not sample_features: raise RuntimeError("No features found. Check pooling_type/selected_layers.")
        aggregated.append(np.concatenate(sample_features))
    return np.asarray(aggregated, dtype=np.float32)

@torch.inference_mode()
def siren_predict(representations, siren_model, device: str, mlp_batch_size: int):
    X = aggregate_features(representations, siren_model["pooling_type"], siren_model["layer_weights"], siren_model["selected_layers"])
    model = siren_model["final_mlp"]
    preds, unsafe_scores = [], []
    for i in range(0, len(X), mlp_batch_size):
        logits = model(torch.from_numpy(X[i:i+mlp_batch_size]).to(device))
        if logits.shape[-1] == 1:
            score = torch.sigmoid(logits.reshape(-1)); pred = (score >= 0.5).long()
        else:
            probs = torch.softmax(logits, dim=-1); score = probs[:, 1]; pred = torch.argmax(probs, dim=-1)
        preds.extend(pred.detach().cpu().numpy().astype(int).tolist())
        unsafe_scores.extend(score.detach().cpu().numpy().astype(float).tolist())
    return np.asarray(preds, dtype=int), np.asarray(unsafe_scores, dtype=float)

def compute_metrics(gold, pred, score):
    out = {"n": int(len(gold)), "accuracy": float(accuracy_score(gold, pred)), "precision_unsafe": float(precision_score(gold, pred, pos_label=1, zero_division=0)), "recall_unsafe": float(recall_score(gold, pred, pos_label=1, zero_division=0)), "f1_unsafe": float(f1_score(gold, pred, pos_label=1, zero_division=0)), "f1_macro": float(f1_score(gold, pred, average="macro", zero_division=0))}
    if len(np.unique(gold)) == 2:
        try:
            out["auroc"] = float(roc_auc_score(gold, score)); out["auprc"] = float(average_precision_score(gold, score))
        except Exception:
            out["auroc"] = None; out["auprc"] = None
    return out

def batched(xs, batch_size):
    for i in range(0, len(xs), batch_size): yield xs[i:i+batch_size]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--pkl_path", required=True); p.add_argument("--output", required=True); p.add_argument("--summary_output", default=None)
    p.add_argument("--evaluator_name", required=True); p.add_argument("--model_key", default=None, choices=list(MODEL_CONFIGS.keys()))
    p.add_argument("--backbone_model_path", default=None); p.add_argument("--langs", nargs="+", default=["en", "ko"], choices=["en", "ko"])
    p.add_argument("--text_mode", choices=["prompt", "response", "prompt_response"], default="prompt_response")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); p.add_argument("--batch_size", type=int, default=8); p.add_argument("--mlp_batch_size", type=int, default=256)
    p.add_argument("--max_length", type=int, default=512); p.add_argument("--torch_dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    p.add_argument("--attn_implementation", default=None); p.add_argument("--trust_remote_code", action="store_true", default=True); p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    backbone_path = args.backbone_model_path or (MODEL_CONFIGS[args.model_key] if args.model_key else None)
    if not backbone_path: raise ValueError("Provide either --model_key or --backbone_model_path")
    rows, out_path = read_jsonl(Path(args.data)), Path(args.output)
    siren_model = load_siren_pickle(Path(args.pkl_path), args.device)
    extractor = RepresentationExtractor(backbone_path, args.device, args.batch_size, args.max_length, args.torch_dtype, args.attn_implementation, args.trust_remote_code)
    extractor.register_hooks(); existing = get_existing_keys(out_path) if args.resume else set(); summary = []
    try:
        for lang in args.langs:
            work_rows = [r for r in rows if (str(r["id"]), args.evaluator_name, lang) not in existing]
            print(f"[eval] evaluator={args.evaluator_name} lang={lang} rows={len(work_rows)} / total={len(rows)}")
            for batch_rows in tqdm(list(batched(work_rows, args.batch_size)), desc=f"{args.evaluator_name}:{lang}"):
                texts = [get_text(r, lang, args.text_mode) for r in batch_rows]
                reps = extractor.extract_batch(texts)
                pred, score = siren_predict(reps, siren_model, args.device, args.mlp_batch_size)
                out_rows = []
                for r, t, yhat, s in zip(batch_rows, texts, pred.tolist(), score.tolist()):
                    gold_int = r.get("gold_int")
                    if gold_int is None: gold_int = label_to_int(r.get("gold_label", r.get("label")))
                    gold_label = "unsafe" if int(gold_int) == 1 else "safe"
                    out_rows.append({
                        "id": r["id"], "source_id": r.get("source_id"), "source_dataset": r.get("source_dataset"), "split": r.get("split"),
                        "evaluator_name": args.evaluator_name, "model_key": args.model_key, "backbone_model_path": backbone_path, "pkl_path": str(args.pkl_path),
                        "lang": lang, "text_mode": args.text_mode, "label": gold_label, "gold_label": gold_label, "gold_int": int(gold_int),
                        "prediction": "unsafe" if int(yhat) == 1 else "safe", "pred_int": int(yhat), "unsafe_score": float(s),
                        "category": r.get("category"), "violated_categories": r.get("violated_categories"), "text_chars": len(t),
                    })
                append_jsonl(out_path, out_rows)
            current = [r for r in read_jsonl(out_path) if r.get("evaluator_name") == args.evaluator_name and r.get("lang") == lang]
            if current:
                gold = np.asarray([int(r["gold_int"]) for r in current]); pred = np.asarray([int(r["pred_int"]) for r in current]); score = np.asarray([float(r["unsafe_score"]) for r in current])
                m = compute_metrics(gold, pred, score); m.update({"evaluator_name": args.evaluator_name, "lang": lang, "text_mode": args.text_mode}); summary.append(m)
                print(f"[metrics] {args.evaluator_name} {lang}: {m}")
    finally:
        extractor.close()
    if args.summary_output:
        Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] predictions -> {out_path}")
if __name__ == "__main__": main()
