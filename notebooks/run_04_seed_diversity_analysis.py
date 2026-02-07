import argparse
import itertools
import json
import math
import re
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.optimize import minimize
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.gru_baseline import GRUBaseline
from src.data.preprocessing import Normalizer, DerivedFeatureBuilder
from utils import weighted_pearson_correlation

RAW_COLS = [f'p{i}' for i in range(12)] + [f'v{i}' for i in range(12)] + [f'dp{i}' for i in range(4)] + [f'dv{i}' for i in range(4)]
TARGET_COLS = ['t0', 't1']


def parse_args():
    parser = argparse.ArgumentParser(description='Seed diversity + ensemble analysis for GRU checkpoints')
    parser.add_argument('--config', type=str, default='configs/gru_derived_tightwd_v2.yaml')
    parser.add_argument('--normalizer', type=str, default='logs/normalizer.npz')
    parser.add_argument('--data', type=str, default='datasets/valid.parquet')
    parser.add_argument('--artifacts', type=str, default='notebooks/artifacts/04_seed_diversity')
    parser.add_argument('--checkpoints', type=str, nargs='*', default=None,
                        help='Explicit checkpoint list. If omitted, uses seed-pattern with --seeds.')
    parser.add_argument('--seeds', type=int, nargs='*', default=[42, 43, 44, 45, 46])
    parser.add_argument('--ckpt-pattern', type=str, default='logs/gru_derived_tightwd_v2_seed{seed}.pt')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--max_rows', type=int, default=None,
                        help='Optional debug cap on rows processed from valid parquet.')
    return parser.parse_args()


def build_checkpoint_list(args) -> List[Path]:
    if args.checkpoints:
        return [ROOT / p for p in args.checkpoints]
    return [ROOT / args.ckpt_pattern.format(seed=s) for s in args.seeds]


def checkpoint_label(path: Path, ckpt_obj: dict, idx: int) -> str:
    seed = ckpt_obj.get('seed', None)
    if seed is None:
        m = re.search(r'seed(\d+)', path.stem)
        if m:
            seed = int(m.group(1))
    if seed is None:
        return f'model_{idx}'
    return f'seed{seed}'


def load_models(checkpoints: List[Path], fallback_config_path: Path, normalizer_path: Path, device: torch.device):
    with fallback_config_path.open('r', encoding='utf-8') as f:
        fallback_cfg = yaml.safe_load(f)

    models = []
    configs = []
    labels = []
    normalizers = []

    for i, ckpt_path in enumerate(checkpoints):
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg = ckpt.get('config', fallback_cfg)
        model_type = cfg.get('model', {}).get('type', 'gru')
        if model_type != 'gru':
            raise ValueError(f'Only GRU checkpoints supported in this analysis. Got {model_type} at {ckpt_path}')

        model = GRUBaseline(cfg)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        model.eval()

        models.append(model)
        configs.append(cfg)
        labels.append(checkpoint_label(ckpt_path, ckpt, i))
        normalizers.append(Normalizer.load(str(normalizer_path)))

    return models, configs, labels, normalizers


def score_target(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(weighted_pearson_correlation(y_true, np.clip(y_pred, -6, 6)))


def score_avg(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    t0 = score_target(y_true[:, 0], y_pred[:, 0])
    t1 = score_target(y_true[:, 1], y_pred[:, 1])
    return {'t0': t0, 't1': t1, 'avg': (t0 + t1) / 2.0}


def infer_all_models_stepwise(models, configs, labels, normalizers, df: pd.DataFrame, device: torch.device):
    seq_ix = df['seq_ix'].to_numpy(np.int64)
    need_pred = df['need_prediction'].to_numpy().astype(bool)
    X_raw = df[RAW_COLS].to_numpy(np.float32)
    y_true_all = df[TARGET_COLS].to_numpy(np.float32)

    n_models = len(models)
    n_rows = len(df)
    n_scored = int(need_pred.sum())

    y_true = np.empty((n_scored, 2), dtype=np.float32)
    preds = np.empty((n_models, n_scored, 2), dtype=np.float32)

    # Pipeline compatibility check: can share feature transform across models
    derived_flags = [bool(c.get('data', {}).get('derived_features', False)) for c in configs]
    share_pipeline = len(set(derived_flags)) == 1

    hiddens = [None] * n_models
    current_seq = None
    out_idx = 0

    with torch.no_grad():
        for i in range(n_rows):
            s = seq_ix[i]
            if current_seq is None or s != current_seq:
                current_seq = s
                hiddens = [None] * n_models

            raw_step = X_raw[i:i+1]  # shape (1, 32)

            if share_pipeline:
                if derived_flags[0]:
                    derived = DerivedFeatureBuilder.compute(raw_step)
                    feat = np.concatenate([raw_step, derived], axis=-1)
                else:
                    feat = raw_step
                feat_norm = normalizers[0].transform(feat)
                x = torch.from_numpy(feat_norm).to(device)

                for m in range(n_models):
                    pred, hiddens[m] = models[m].forward_step(x, hiddens[m])
                    pred_np = np.clip(pred.squeeze(0).cpu().numpy(), -6, 6)
                    if need_pred[i]:
                        preds[m, out_idx] = pred_np
            else:
                for m in range(n_models):
                    if derived_flags[m]:
                        derived = DerivedFeatureBuilder.compute(raw_step)
                        feat = np.concatenate([raw_step, derived], axis=-1)
                    else:
                        feat = raw_step
                    feat_norm = normalizers[m].transform(feat)
                    x = torch.from_numpy(feat_norm).to(device)
                    pred, hiddens[m] = models[m].forward_step(x, hiddens[m])
                    pred_np = np.clip(pred.squeeze(0).cpu().numpy(), -6, 6)
                    if need_pred[i]:
                        preds[m, out_idx] = pred_np

            if need_pred[i]:
                y_true[out_idx] = y_true_all[i]
                out_idx += 1

    return preds, y_true


def pairwise_corr_matrix(preds: np.ndarray, target_idx: int = None) -> np.ndarray:
    n_models = preds.shape[0]
    mat = np.eye(n_models, dtype=np.float64)
    for i in range(n_models):
        for j in range(i + 1, n_models):
            if target_idx is None:
                a = preds[i].reshape(-1)
                b = preds[j].reshape(-1)
            else:
                a = preds[i, :, target_idx]
                b = preds[j, :, target_idx]
            c = float(np.corrcoef(a, b)[0, 1])
            mat[i, j] = c
            mat[j, i] = c
    return mat


def optimize_weights(preds: np.ndarray, y_true: np.ndarray):
    n_models = preds.shape[0]

    def objective(w):
        y = np.tensordot(w, preds, axes=(0, 0))
        s = score_avg(y_true, y)['avg']
        return -s

    x0 = np.ones(n_models, dtype=np.float64) / n_models
    bounds = [(0.0, 1.0)] * n_models
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)

    result = minimize(
        objective,
        x0=x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 400, 'ftol': 1e-10, 'disp': False},
    )

    w = result.x
    w = np.clip(w, 0.0, 1.0)
    w = w / (w.sum() + 1e-12)
    y_opt = np.tensordot(w, preds, axes=(0, 0))
    scores = score_avg(y_true, y_opt)

    return {
        'weights': w,
        'scores': scores,
        'success': bool(result.success),
        'message': str(result.message),
        'nit': int(result.nit),
    }


def best_subset_scores(preds: np.ndarray, y_true: np.ndarray, labels: List[str]) -> pd.DataFrame:
    n_models = preds.shape[0]
    rows = []
    for n in range(1, n_models + 1):
        best = None
        for subset in itertools.combinations(range(n_models), n):
            y = preds[list(subset)].mean(axis=0)
            sc = score_avg(y_true, y)
            row = {
                'n_models': n,
                'subset_indices': list(subset),
                'subset_labels': [labels[k] for k in subset],
                'score_t0': sc['t0'],
                'score_t1': sc['t1'],
                'score_avg': sc['avg'],
            }
            if best is None or row['score_avg'] > best['score_avg']:
                best = row
        rows.append(best)
    return pd.DataFrame(rows)


def diversity_by_target_bucket(preds: np.ndarray, y_true: np.ndarray) -> pd.DataFrame:
    rows = []
    n_models = preds.shape[0]

    for t_idx, t_name in enumerate(['t0', 't1']):
        y_abs = np.abs(y_true[:, t_idx])
        quantiles = np.quantile(y_abs, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        for b in range(5):
            lo, hi = quantiles[b], quantiles[b + 1]
            if b < 4:
                mask = (y_abs >= lo) & (y_abs < hi)
            else:
                mask = (y_abs >= lo) & (y_abs <= hi)

            bucket_preds = preds[:, mask, t_idx]
            if bucket_preds.shape[1] == 0:
                continue

            # Mean per-sample disagreement across models
            sample_std = bucket_preds.std(axis=0)
            sample_range = bucket_preds.max(axis=0) - bucket_preds.min(axis=0)

            # Mean pairwise corr inside bucket
            pair_corr = []
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    c = float(np.corrcoef(bucket_preds[i], bucket_preds[j])[0, 1])
                    pair_corr.append(c)

            rows.append({
                'target': t_name,
                'bucket': b + 1,
                'abs_target_min': float(lo),
                'abs_target_max': float(hi),
                'n_samples': int(mask.sum()),
                'mean_abs_target': float(y_abs[mask].mean()),
                'mean_pred_std': float(sample_std.mean()),
                'p90_pred_std': float(np.quantile(sample_std, 0.9)),
                'mean_pred_range': float(sample_range.mean()),
                'mean_pairwise_corr': float(np.mean(pair_corr)) if pair_corr else 1.0,
            })

    return pd.DataFrame(rows)


def save_heatmap(matrix: np.ndarray, labels: List[str], title: str, path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap='viridis', vmin=min(0.8, float(np.nanmin(matrix))), vmax=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_title(title)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center', color='white', fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()

    artifacts = ROOT / args.artifacts
    artifacts.mkdir(parents=True, exist_ok=True)

    checkpoints = build_checkpoint_list(args)
    missing = [str(p) for p in checkpoints if not p.exists()]
    if missing:
        # Write placeholder artifacts with expected schemas for smooth handoff.
        pd.DataFrame(columns=['model', 'score_t0', 'score_t1', 'score_avg']).to_csv(
            artifacts / 'model_scores.csv', index=False
        )
        pd.DataFrame(columns=['model_i', 'model_j', 'corr_flat', 'corr_t0', 'corr_t1']).to_csv(
            artifacts / 'pairwise_correlations.csv', index=False
        )
        pd.DataFrame(columns=['n_models', 'subset_indices', 'subset_labels', 'score_t0', 'score_t1', 'score_avg']).to_csv(
            artifacts / 'ensemble_vs_n_models.csv', index=False
        )
        pd.DataFrame(columns=[
            'target', 'bucket', 'abs_target_min', 'abs_target_max', 'n_samples',
            'mean_abs_target', 'mean_pred_std', 'p90_pred_std', 'mean_pred_range', 'mean_pairwise_corr'
        ]).to_csv(artifacts / 'diversity_by_target_bucket.csv', index=False)
        with (artifacts / 'optimal_weights.json').open('w', encoding='utf-8') as f:
            json.dump({'status': 'pending_checkpoints', 'models': [], 'weights': []}, f, indent=2)

        msg = {
            'status': 'missing_checkpoints',
            'expected': [str(p) for p in checkpoints],
            'missing': missing,
            'next_step': 'Train or copy the seed checkpoints, then rerun notebooks/run_04_seed_diversity_analysis.py',
        }
        with (artifacts / 'status.json').open('w', encoding='utf-8') as f:
            json.dump(msg, f, indent=2)
        raise FileNotFoundError('Missing checkpoints: ' + ', '.join(missing))

    normalizer_path = ROOT / args.normalizer
    if not normalizer_path.exists():
        raise FileNotFoundError(f'Normalizer not found: {normalizer_path}')

    data_path = ROOT / args.data
    if not data_path.exists():
        raise FileNotFoundError(f'Dataset not found: {data_path}')

    device = torch.device(args.device)

    print('Loading models...')
    models, configs, labels, normalizers = load_models(
        checkpoints=checkpoints,
        fallback_config_path=ROOT / args.config,
        normalizer_path=normalizer_path,
        device=device,
    )
    print('Models:', labels)

    print('Loading valid parquet...')
    df = pd.read_parquet(data_path)
    df = df.sort_values(['seq_ix', 'step_in_seq']).reset_index(drop=True)
    if args.max_rows is not None:
        df = df.iloc[:args.max_rows].copy()
        print(f'Using debug subset rows: {len(df)}')

    print('Running step-by-step inference across all models...')
    preds, y_true = infer_all_models_stepwise(models, configs, labels, normalizers, df, device)

    # Per-model scores
    score_rows = []
    for i, label in enumerate(labels):
        sc = score_avg(y_true, preds[i])
        score_rows.append({'model': label, 'score_t0': sc['t0'], 'score_t1': sc['t1'], 'score_avg': sc['avg']})
    model_scores = pd.DataFrame(score_rows).sort_values('score_avg', ascending=False)
    model_scores.to_csv(artifacts / 'model_scores.csv', index=False)

    # Pairwise correlations
    corr_flat = pairwise_corr_matrix(preds, target_idx=None)
    corr_t0 = pairwise_corr_matrix(preds, target_idx=0)
    corr_t1 = pairwise_corr_matrix(preds, target_idx=1)

    pair_rows = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            pair_rows.append({
                'model_i': labels[i],
                'model_j': labels[j],
                'corr_flat': float(corr_flat[i, j]),
                'corr_t0': float(corr_t0[i, j]),
                'corr_t1': float(corr_t1[i, j]),
            })
    pairwise = pd.DataFrame(pair_rows)
    pairwise.to_csv(artifacts / 'pairwise_correlations.csv', index=False)

    # Ensemble average
    avg_pred = preds.mean(axis=0)
    avg_scores = score_avg(y_true, avg_pred)

    # Optimal weights
    opt = optimize_weights(preds, y_true)
    opt_payload = {
        'models': labels,
        'weights': [float(x) for x in opt['weights']],
        'scores': {k: float(v) for k, v in opt['scores'].items()},
        'uniform_average_scores': {k: float(v) for k, v in avg_scores.items()},
        'optimizer_success': opt['success'],
        'optimizer_message': opt['message'],
        'optimizer_iterations': int(opt['nit']),
        'n_models': int(len(labels)),
        'n_scored_samples': int(len(y_true)),
    }
    with (artifacts / 'optimal_weights.json').open('w', encoding='utf-8') as f:
        json.dump(opt_payload, f, indent=2)

    # Ensemble vs N
    ens_n = best_subset_scores(preds, y_true, labels)
    ens_n.to_csv(artifacts / 'ensemble_vs_n_models.csv', index=False)

    # Diversity by |target| bucket
    div_bucket = diversity_by_target_bucket(preds, y_true)
    div_bucket.to_csv(artifacts / 'diversity_by_target_bucket.csv', index=False)

    # Plots
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(model_scores))
    width = 0.25
    ax.bar(x - width, model_scores['score_t0'], width=width, label='t0')
    ax.bar(x, model_scores['score_t1'], width=width, label='t1')
    ax.bar(x + width, model_scores['score_avg'], width=width, label='avg')
    ax.set_xticks(x)
    ax.set_xticklabels(model_scores['model'], rotation=45, ha='right')
    ax.set_ylabel('Weighted Pearson')
    ax.set_title('Per-Model Validation Scores')
    ax.legend()
    fig.tight_layout()
    fig.savefig(artifacts / 'per_model_scores.png', dpi=180)
    plt.close(fig)

    save_heatmap(corr_flat, labels, 'Pairwise Prediction Correlation (flattened t0+t1)', artifacts / 'pairwise_corr_heatmap.png')

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ens_n['n_models'], ens_n['score_avg'], marker='o', label='avg')
    ax.plot(ens_n['n_models'], ens_n['score_t0'], marker='o', linestyle='--', label='t0')
    ax.plot(ens_n['n_models'], ens_n['score_t1'], marker='o', linestyle='--', label='t1')
    ax.set_xlabel('Number of models in best subset')
    ax.set_ylabel('Weighted Pearson')
    ax.set_title('Ensemble Score vs Number of Models')
    ax.set_xticks(ens_n['n_models'])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(artifacts / 'ensemble_vs_n_models.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for ax_i, t_name in enumerate(['t0', 't1']):
        d = div_bucket[div_bucket['target'] == t_name].sort_values('bucket')
        axes[ax_i].plot(d['bucket'], d['mean_pred_std'], marker='o', label='mean_pred_std')
        axes[ax_i].plot(d['bucket'], d['mean_pred_range'], marker='o', label='mean_pred_range')
        axes[ax_i].set_title(f'Diversity by |{t_name}| Quintile')
        axes[ax_i].set_xlabel('Quintile (low->high |target|)')
        axes[ax_i].set_xticks([1, 2, 3, 4, 5])
        axes[ax_i].grid(True, alpha=0.3)
    axes[0].set_ylabel('Disagreement magnitude')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(artifacts / 'diversity_by_target_bucket.png', dpi=180)
    plt.close(fig)

    print('Analysis complete. Artifacts written to', artifacts)


if __name__ == '__main__':
    main()
