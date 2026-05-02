#!/usr/bin/env python3
"""Inspect a SIREN best_model.pkl file."""
from __future__ import annotations
import argparse, pickle
from pathlib import Path
import torch.nn as nn
class AdaptiveMLPClassifier(nn.Module):
    def __init__(self, input_dim, layer_dims, dropout_rates, num_classes=2):
        super().__init__(); layers=[]; prev=input_dim
        for h,d in zip(layer_dims, dropout_rates):
            layers += [nn.Linear(prev,h), nn.ReLU()]
            if d>0: layers.append(nn.Dropout(d))
            prev=h
        layers.append(nn.Linear(prev,num_classes)); self.network=nn.Sequential(*layers)
    def forward(self,x): return self.network(x)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--pkl_path', required=True); args=p.parse_args()
    path=Path(args.pkl_path)
    with path.open('rb') as f: obj=pickle.load(f)
    print(f'path: {path}'); print(f'type: {type(obj)}')
    if isinstance(obj, dict):
        print('keys:')
        for k,v in obj.items(): print(f'  - {k}: {type(v)}')
        if 'pooling_type' in obj: print('pooling_type:', obj['pooling_type'])
        if 'selected_layers' in obj: print('selected_layers:', obj['selected_layers'])
        if 'layer_weights' in obj: print('layer_weights keys:', list(obj['layer_weights'].keys())[:10], '...')
        if 'threshold' in obj: print('threshold:', obj['threshold'])
        if 'selected_neurons_dict' in obj:
            snd=obj['selected_neurons_dict']; print('selected_neurons_dict total:', len(snd))
            for i,(k,v) in enumerate(snd.items()):
                print(f'  {k}: n={len(v)} first={list(v)[:5]}')
                if i>=4: break
        if 'final_mlp' in obj:
            print(obj['final_mlp']); print('final_mlp params:', sum(p.numel() for p in obj['final_mlp'].parameters()))
if __name__=='__main__': main()
