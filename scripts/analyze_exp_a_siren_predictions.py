#!/usr/bin/env python3
"""Analyze Exp-A SIREN prediction JSONL."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, List
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

def read_jsonl(path: Path) -> List[dict]:
    with path.open('r', encoding='utf-8') as f: return [json.loads(line) for line in f if line.strip()]
def normalize_label(x: Any):
    if x is None: return None
    s=str(x).strip().lower()
    if s in {'safe','s','0','false','unharmful','acceptable','benign'}: return 'safe'
    if s in {'unsafe','u','1','true','harmful','unacceptable'}: return 'unsafe'
    if s.startswith('safe'): return 'safe'
    if s.startswith('unsafe'): return 'unsafe'
    return None
def load_df(paths: List[str]) -> pd.DataFrame:
    rows=[]
    for p in paths:
        for r in read_jsonl(Path(p)):
            gold=normalize_label(r.get('gold_label', r.get('label'))); pred=normalize_label(r.get('prediction'))
            rows.append({'id':str(r.get('id')), 'evaluator_name':r.get('evaluator_name', r.get('evaluator','unknown')), 'lang':str(r.get('lang')).lower(), 'gold_label':gold, 'prediction':pred, 'gold_int':1 if gold=='unsafe' else 0 if gold=='safe' else None, 'pred_int':1 if pred=='unsafe' else 0 if pred=='safe' else None, 'unsafe_score':r.get('unsafe_score'), 'category':r.get('category'), 'text_mode':r.get('text_mode')})
    df=pd.DataFrame(rows)
    if df.empty: raise RuntimeError('No prediction rows loaded.')
    df['unsafe_score']=pd.to_numeric(df['unsafe_score'], errors='coerce')
    return df
def metrics(df: pd.DataFrame) -> pd.DataFrame:
    out=[]
    for (ev,lang),g in df.dropna(subset=['gold_int','pred_int']).groupby(['evaluator_name','lang']):
        y=g['gold_int'].astype(int); p=g['pred_int'].astype(int)
        row={'evaluator_name':ev,'lang':lang,'n':len(g),'accuracy':accuracy_score(y,p),'precision_unsafe':precision_score(y,p,pos_label=1,zero_division=0),'recall_unsafe':recall_score(y,p,pos_label=1,zero_division=0),'f1_unsafe':f1_score(y,p,pos_label=1,zero_division=0),'f1_macro':f1_score(y,p,average='macro',zero_division=0)}
        scored=g.dropna(subset=['unsafe_score'])
        if len(scored)>=2 and scored['gold_int'].nunique()==2:
            try:
                row['auroc']=roc_auc_score(scored['gold_int'].astype(int), scored['unsafe_score'].astype(float)); row['auprc']=average_precision_score(scored['gold_int'].astype(int), scored['unsafe_score'].astype(float))
            except Exception: row['auroc']=None; row['auprc']=None
        out.append(row)
    return pd.DataFrame(out).sort_values(['evaluator_name','lang'])
def consistency(df: pd.DataFrame) -> pd.DataFrame:
    out=[]; valid=df.dropna(subset=['id','evaluator_name','lang','prediction'])
    for ev,g in valid.groupby('evaluator_name'):
        piv=g.pivot_table(index='id', columns='lang', values='prediction', aggfunc='first')
        if 'en' not in piv.columns or 'ko' not in piv.columns: continue
        paired=piv.dropna(subset=['en','ko'])
        row={'evaluator_name':ev,'n_pairs':len(paired),'consistency':(paired['en']==paired['ko']).mean(),'flip_rate':(paired['en']!=paired['ko']).mean(),'en_safe_to_ko_unsafe':((paired['en']=='safe')&(paired['ko']=='unsafe')).mean(),'en_unsafe_to_ko_safe':((paired['en']=='unsafe')&(paired['ko']=='safe')).mean()}
        spiv=g.pivot_table(index='id', columns='lang', values='unsafe_score', aggfunc='first')
        if 'en' in spiv.columns and 'ko' in spiv.columns:
            sp=spiv.dropna(subset=['en','ko'])
            if not sp.empty:
                row['mean_abs_score_gap']=(sp['en']-sp['ko']).abs().mean(); row['mean_score_gap_ko_minus_en']=(sp['ko']-sp['en']).mean()
        out.append(row)
    return pd.DataFrame(out).sort_values('evaluator_name')
def main():
    p=argparse.ArgumentParser(); p.add_argument('--predictions', nargs='+', required=True); p.add_argument('--out_dir', default='results/exp_a/analysis'); args=p.parse_args()
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df=load_df(args.predictions); df.to_csv(out_dir/'normalized_predictions.csv', index=False)
    m=metrics(df); c=consistency(df)
    m.to_csv(out_dir/'metrics_by_evaluator_lang.csv', index=False); c.to_csv(out_dir/'language_consistency.csv', index=False)
    print('\n=== Metrics ==='); print(m.to_string(index=False)); print('\n=== EN/KO consistency ==='); print(c.to_string(index=False)); print(f'\n[done] {out_dir}')
if __name__=='__main__': main()
