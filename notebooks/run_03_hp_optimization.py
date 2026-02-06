import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
ART_DIR = ROOT / 'notebooks' / 'artifacts' / '03_hp_optimization'
ART_DIR.mkdir(parents=True, exist_ok=True)

EXP_PATH = ROOT / 'logs' / 'experiments.jsonl'

# Experiments to analyze (from findings table)
TARGET_CONFIGS = [
    'configs/gru_baseline.yaml',
    'configs/gru_derived_v1.yaml',
    'configs/gru_derived_t1focus_v1.yaml',
    'configs/gru_long_memory_derived_v1.yaml',
    'configs/lstm_derived_v1.yaml',
]


def as_float(v):
    try:
        return float(v)
    except Exception:
        return np.nan


def read_jsonl(path: Path) -> list:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_param_count(notes: str):
    if not isinstance(notes, str):
        return np.nan
    m = re.search(r'(\d+)K params', notes)
    if m:
        return int(m.group(1)) * 1000
    return np.nan


rows = read_jsonl(EXP_PATH)
df = pd.DataFrame(rows)

# Keep experiments that have score info and belong to target config set
df = df[df['config'].isin(TARGET_CONFIGS)].copy()
df = df[df['val_score_avg'].notna()].copy()

# If duplicate run exists for a config, keep all for variability stats,
# but also create a best-run table for per-config comparison.
df['param_count_from_notes'] = df['notes'].apply(parse_param_count)
df['best_epoch'] = df['best_epoch'].astype(float)
df['epochs_trained'] = df['epochs_trained'].astype(float)
df['val_score_avg'] = df['val_score_avg'].astype(float)
df['val_score_t0'] = df['val_score_t0'].astype(float)
df['val_score_t1'] = df['val_score_t1'].astype(float)

# Merge in config hyperparameters
cfg_rows = []
for cfg in sorted(set(df['config'].tolist())):
    p = ROOT / cfg
    if p.exists():
        with p.open('r', encoding='utf-8') as f:
            c = yaml.safe_load(f)
        cfg_rows.append({
            'config': cfg,
            'model_type': c.get('model', {}).get('type'),
            'input_size': c.get('model', {}).get('input_size'),
            'hidden_size': c.get('model', {}).get('hidden_size'),
            'num_layers': c.get('model', {}).get('num_layers'),
            'dropout': as_float(c.get('model', {}).get('dropout')),
            'lr': as_float(c.get('training', {}).get('lr')),
            'weight_decay': as_float(c.get('training', {}).get('weight_decay')),
            'batch_size': c.get('training', {}).get('batch_size'),
            'weighted_ratio': as_float(c.get('training', {}).get('weighted_ratio')),
            'target_weights': str(c.get('training', {}).get('target_weights')),
            'early_stopping_patience': c.get('training', {}).get('early_stopping_patience'),
            'derived_features': c.get('data', {}).get('derived_features', False),
            'scheduler_type': c.get('training', {}).get('scheduler', {}).get('type', 'reduce_on_plateau'),
        })

cfg_df = pd.DataFrame(cfg_rows)
df = df.merge(cfg_df, on='config', how='left')

# Best run per config for direct comparison
best_per_cfg = (
    df.sort_values('val_score_avg', ascending=False)
      .groupby('config', as_index=False)
      .first()
      .sort_values('val_score_avg', ascending=False)
)

best_per_cfg.to_csv(ART_DIR / 'experiment_comparison_best_runs.csv', index=False)
df.to_csv(ART_DIR / 'experiment_comparison_all_runs.csv', index=False)

# Stopping dynamics metrics (proxy for training curve shape)
best_per_cfg['overfit_tail_epochs'] = best_per_cfg['epochs_trained'] - best_per_cfg['best_epoch']
best_per_cfg['overfit_tail_ratio'] = best_per_cfg['overfit_tail_epochs'] / best_per_cfg['epochs_trained'].clip(lower=1)
best_per_cfg['early_peak_flag'] = best_per_cfg['best_epoch'] <= 11

best_per_cfg[[
    'config', 'val_score_avg', 'best_epoch', 'epochs_trained',
    'overfit_tail_epochs', 'overfit_tail_ratio', 'lr', 'dropout',
    'batch_size', 'weight_decay', 'hidden_size', 'num_layers', 'model_type'
]].to_csv(ART_DIR / 'stopping_dynamics_summary.csv', index=False)

# Plot stopping dynamics
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_df = best_per_cfg.sort_values('val_score_avg', ascending=False)
axes[0].bar(plot_df['config'].str.replace('configs/', '', regex=False), plot_df['best_epoch'], label='best_epoch', alpha=0.9)
axes[0].bar(plot_df['config'].str.replace('configs/', '', regex=False), plot_df['epochs_trained'] - plot_df['best_epoch'],
            bottom=plot_df['best_epoch'], label='post-peak tail', alpha=0.6)
axes[0].set_title('Stopping Dynamics by Experiment')
axes[0].set_ylabel('Epochs')
axes[0].tick_params(axis='x', rotation=35)
axes[0].legend()

axes[1].scatter(plot_df['best_epoch'], plot_df['val_score_avg'], s=120, c='#1f77b4')
for _, r in plot_df.iterrows():
    axes[1].annotate(r['config'].replace('configs/', '').replace('.yaml', ''),
                     (r['best_epoch'], r['val_score_avg']),
                     textcoords='offset points', xytext=(5, 3), fontsize=8)
axes[1].set_title('Best Epoch vs Validation Avg')
axes[1].set_xlabel('Best epoch')
axes[1].set_ylabel('Val score avg')

plt.tight_layout()
plt.savefig(ART_DIR / 'stopping_dynamics.png', dpi=180)
plt.close(fig)

# Hyperparameter sensitivity (coarse, from limited experiments)
hp_cols = ['lr', 'dropout', 'batch_size', 'weight_decay']
sensitivity_rows = []
for c in hp_cols:
    if c in best_per_cfg and best_per_cfg[c].notna().sum() >= 3:
        x = best_per_cfg[c].astype(float).to_numpy()
        y = best_per_cfg['val_score_avg'].to_numpy()
        corr = np.corrcoef(x, y)[0, 1]
        sensitivity_rows.append({'hyperparam': c, 'pearson_with_val_avg': float(corr)})

sens_df = pd.DataFrame(sensitivity_rows)
sens_df.to_csv(ART_DIR / 'hyperparam_sensitivity_coarse.csv', index=False)

# Recommended sweep ranges based on top runs + stopping dynamics
# Use only GRU+derived runs for actionable sweep guidance
gru_derived = best_per_cfg[(best_per_cfg['model_type'] == 'gru') & (best_per_cfg['derived_features'] == True)].copy()
gru_derived = gru_derived.sort_values('val_score_avg', ascending=False)

# top configs near best
anchor = gru_derived.head(min(3, len(gru_derived)))

# fallback values
lr_center = float(anchor['lr'].median()) if len(anchor) else 0.001
drop_center = float(anchor['dropout'].median()) if len(anchor) else 0.2
bs_center = int(anchor['batch_size'].median()) if len(anchor) else 256
wd_center = float(anchor['weight_decay'].median()) if len(anchor) else 1e-5

reco = {
    'lr': {
        'recommended_range': [max(3e-4, lr_center * 0.7), min(2e-3, lr_center * 1.5)],
        'grid_hint': [0.0006, 0.0008, 0.0010, 0.0012, 0.0015],
    },
    'dropout': {
        'recommended_range': [max(0.12, drop_center - 0.05), min(0.35, drop_center + 0.08)],
        'grid_hint': [0.15, 0.2, 0.24, 0.28],
    },
    'batch_size': {
        'recommended_range': [128, 320],
        'grid_hint': [160, 192, 224, 256, 320],
    },
    'weight_decay': {
        'recommended_range': [1e-5, 8e-5],
        'grid_hint': [1e-5, 2e-5, 4e-5, 6e-5, 8e-5],
    },
    'epochs': {
        'recommended_max': 35,
        'best_epoch_observed_range': [int(best_per_cfg['best_epoch'].min()), int(best_per_cfg['best_epoch'].max())],
        'note': 'Keep early stopping; most value appears by epochs 4-11.'
    },
}

with open(ART_DIR / 'recommended_hp_ranges.json', 'w', encoding='utf-8') as f:
    json.dump(reco, f, indent=2)

# Regularization recommendations table
reg_table = pd.DataFrame([
    {
        'technique': 'label_smoothing',
        'recommendation': 'Not recommended as primary method',
        'reason': 'This is continuous regression; label smoothing is mainly classification-oriented and can damp useful amplitude signal.',
    },
    {
        'technique': 'mixup_sequence',
        'recommendation': 'Use cautiously (low priority)',
        'reason': 'Can regularize but may distort temporal microstructure; only consider sequence-level mixup with low alpha and strict validation checks.',
    },
    {
        'technique': 'feature_dropout',
        'recommendation': 'Recommended',
        'reason': 'Likely robust to train-valid feature shift; try random masking/noise on input channels during training (small probability).',
    },
    {
        'technique': 'higher_weight_decay',
        'recommendation': 'Recommended',
        'reason': 'Fast early peaks indicate overfitting; moderate weight decay increases post-peak robustness.',
    },
    {
        'technique': 'dropout_tuning',
        'recommendation': 'Recommended',
        'reason': 'Current best uses 0.2; moderate increases (0.22-0.28) may reduce overfit without hurting fit speed.',
    },
])
reg_table.to_csv(ART_DIR / 'regularization_recommendations.csv', index=False)

# Curve availability note
curve_note = (
    'Per-epoch train/val curves are not persisted in logs/experiments.jsonl or checkpoints for the 5 runs. '\
    'Analysis therefore uses best_epoch, epochs_trained, and score outcomes as curve proxies. '\
    'For future sweeps, persist per-epoch history to enable exact curve diagnostics.'
)
with open(ART_DIR / 'curve_data_availability.txt', 'w', encoding='utf-8') as f:
    f.write(curve_note + '\n')

summary = {
    'n_runs_analyzed': int(len(best_per_cfg)),
    'best_config': best_per_cfg.iloc[0]['config'] if len(best_per_cfg) else None,
    'best_val_avg': float(best_per_cfg.iloc[0]['val_score_avg']) if len(best_per_cfg) else None,
    'early_stopping_epoch_range': [
        int(best_per_cfg['best_epoch'].min()), int(best_per_cfg['best_epoch'].max())
    ] if len(best_per_cfg) else None,
    'early_peak_configs': best_per_cfg[best_per_cfg['early_peak_flag']]['config'].tolist(),
    'recommended_ranges': reco,
    'curve_data_note': curve_note,
}

with open(ART_DIR / 'hp_optimization_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

print('HP optimization analysis complete. Artifacts saved to', ART_DIR)
