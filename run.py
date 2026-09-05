"""Five-fold evaluation; no tuning on evaluation folds."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from dl_flgl import Config, DLFLGL
from dl_flgl.data import load_dataset, demo_data
from dl_flgl.metrics import evaluate

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--data', type=Path)
    source.add_argument('--demo', action='store_true')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', type=Path, default=Path('outputs/evaluation.json'))
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    config = Config(**(json.loads(args.config.read_text(encoding='utf-8')) if args.config else {}))
    if args.device is not None:
        config = replace(config, device=args.device)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    x, y = load_dataset(args.data) if args.data else demo_data(config.seed)
    if not 2 <= args.folds <= len(x):
        parser.error('Require 2 <= folds <= sample count')
    rows = []
    for fold, (train, test) in enumerate(KFold(args.folds, shuffle=True, random_state=config.seed).split(x), 1):
        scaler = MinMaxScaler().fit(x[train])
        model = DLFLGL(config).fit(scaler.transform(x[train]), y[train])
        scores = model.decision_function(scaler.transform(x[test]))
        metrics = evaluate(y[test], scores)
        row = {'fold': fold, 'metrics': metrics, 'train_indices': train.tolist(), 'test_indices': test.tolist(),
               'scaler_min': scaler.data_min_.tolist(), 'scaler_max': scaler.data_max_.tolist(),
               'history': model.history_, 'rules': model.antecedent_.selected_}
        rows.append(row)
        print(json.dumps({'fold': fold, **metrics}), flush=True)
    summary = {}
    for metric in rows[0]['metrics']:
        values = [row['metrics'][metric] for row in rows if row['metrics'][metric] is not None]
        summary[metric] = {'mean': float(np.mean(values)) if values else None,
                           'std': float(np.std(values, ddof=1)) if len(values) > 1 else None,
                           'valid_folds': len(values)}
    report = {'purpose': 'Core implementation validation; not reproduction of manuscript tables',
              'dataset': args.data.name if args.data else 'synthetic demo',
              'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest() if args.data else None,
              'shape': {'X': list(x.shape), 'Y': list(y.shape)}, 'config': asdict(config),
              'torch': torch.__version__, 'summary': summary, 'folds': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
