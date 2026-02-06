import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, spearmanr

ROOT = Path('.')
ART_DIR = ROOT / 'notebooks' / 'artifacts' / '01_eda'
ART_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = ROOT / 'datasets' / 'train.parquet'
VALID_PATH = ROOT / 'datasets' / 'valid.parquet'

FEATURE_COLS = [f'p{i}' for i in range(12)] + [f'v{i}' for i in range(12)] + [f'dp{i}' for i in range(4)] + [f'dv{i}' for i in range(4)]
TARGET_COLS = ['t0', 't1']

RNG = np.random.default_rng(42)


def save_json(obj, path: Path):
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


print('Loading parquet files...')
train = pd.read_parquet(TRAIN_PATH)
valid = pd.read_parquet(VALID_PATH)
print('Train shape:', train.shape)
print('Valid shape:', valid.shape)

# Ensure deterministic ordering for sequence reshaping
def sort_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(['seq_ix', 'step_in_seq']).reset_index(drop=True)

train = sort_df(train)
valid = sort_df(valid)

summary = {}

# Section 1: Data Shape & Basics
print('Section 1: basics...')
section1 = {}
section1['train_shape'] = list(train.shape)
section1['valid_shape'] = list(valid.shape)
section1['n_train_sequences'] = int(train['seq_ix'].nunique())
section1['n_valid_sequences'] = int(valid['seq_ix'].nunique())

train_steps = train.groupby('seq_ix')['step_in_seq'].count()
valid_steps = valid.groupby('seq_ix')['step_in_seq'].count()
section1['train_steps_per_seq_min'] = int(train_steps.min())
section1['train_steps_per_seq_max'] = int(train_steps.max())
section1['valid_steps_per_seq_min'] = int(valid_steps.min())
section1['valid_steps_per_seq_max'] = int(valid_steps.max())

nan_counts_train = train.isna().sum()
nan_counts_valid = valid.isna().sum()
section1['nan_total_train'] = int(nan_counts_train.sum())
section1['nan_total_valid'] = int(nan_counts_valid.sum())
section1['nan_cols_train_nonzero'] = {k: int(v) for k, v in nan_counts_train[nan_counts_train > 0].to_dict().items()}
section1['nan_cols_valid_nonzero'] = {k: int(v) for k, v in nan_counts_valid[nan_counts_valid > 0].to_dict().items()}

numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
inf_train = np.isinf(train[numeric_cols].to_numpy()).sum()
inf_valid = np.isinf(valid[numeric_cols].to_numpy()).sum()
section1['inf_total_train'] = int(inf_train)
section1['inf_total_valid'] = int(inf_valid)
section1['dtypes'] = {c: str(t) for c, t in train.dtypes.items()}

summary['section1'] = section1

# Section 2: Feature Distributions
print('Section 2: distributions...')
section2 = {}

train_feat = train[FEATURE_COLS]
valid_feat = valid[FEATURE_COLS]

train_stats = pd.DataFrame({
    'mean': train_feat.mean(),
    'std': train_feat.std(),
    'min': train_feat.min(),
    'max': train_feat.max(),
    'skew': train_feat.skew(),
    'kurtosis': train_feat.kurtosis(),
})
valid_stats = pd.DataFrame({
    'mean': valid_feat.mean(),
    'std': valid_feat.std(),
    'min': valid_feat.min(),
    'max': valid_feat.max(),
    'skew': valid_feat.skew(),
    'kurtosis': valid_feat.kurtosis(),
})

train_stats.to_csv(ART_DIR / 'feature_summary_train.csv')
valid_stats.to_csv(ART_DIR / 'feature_summary_valid.csv')

# Distribution shift diagnostics using sampled KS test + standardized mean shift
sample_n = 200_000
shift_rows = []
for col in FEATURE_COLS:
    tvals = train[col].to_numpy()
    vvals = valid[col].to_numpy()
    t_idx = RNG.choice(len(tvals), size=min(sample_n, len(tvals)), replace=False)
    v_idx = RNG.choice(len(vvals), size=min(sample_n, len(vvals)), replace=False)
    t_sample = tvals[t_idx]
    v_sample = vvals[v_idx]
    ks_stat, ks_p = ks_2samp(t_sample, v_sample)
    std_train = train_stats.loc[col, 'std']
    mean_shift_z = abs(train_stats.loc[col, 'mean'] - valid_stats.loc[col, 'mean']) / (std_train + 1e-8)
    shift_rows.append({
        'feature': col,
        'ks_stat': ks_stat,
        'ks_pvalue': ks_p,
        'mean_shift_z': mean_shift_z,
    })

shift_df = pd.DataFrame(shift_rows).sort_values('ks_stat', ascending=False)
shift_df.to_csv(ART_DIR / 'feature_shift_train_vs_valid.csv', index=False)

# Feature histogram grid (train vs valid), sampled for speed
hist_n = 150_000
fig, axes = plt.subplots(8, 4, figsize=(20, 30))
axes = axes.flatten()
for i, col in enumerate(FEATURE_COLS):
    ax = axes[i]
    tvals = train[col].to_numpy()
    vvals = valid[col].to_numpy()
    t_sample = tvals[RNG.choice(len(tvals), size=min(hist_n, len(tvals)), replace=False)]
    v_sample = vvals[RNG.choice(len(vvals), size=min(hist_n, len(vvals)), replace=False)]
    lo = float(np.percentile(np.concatenate([t_sample, v_sample]), 0.5))
    hi = float(np.percentile(np.concatenate([t_sample, v_sample]), 99.5))
    bins = np.linspace(lo, hi, 60)
    ax.hist(t_sample, bins=bins, density=True, alpha=0.5, label='train', color='#1f77b4')
    ax.hist(v_sample, bins=bins, density=True, alpha=0.5, label='valid', color='#ff7f0e')
    ax.set_title(col)
    if i == 0:
        ax.legend(fontsize=8)
for j in range(len(FEATURE_COLS), len(axes)):
    axes[j].axis('off')
fig.suptitle('Feature Distributions: Train vs Valid (Sampled)', fontsize=16)
fig.tight_layout()
fig.savefig(ART_DIR / 'feature_hist_train_vs_valid.png', dpi=160)
plt.close(fig)

std_ratio = float(train_stats['std'].max() / (train_stats['std'].min() + 1e-12))
section2['feature_std_max_over_min'] = std_ratio
section2['largest_shift_features'] = shift_df.head(10).to_dict(orient='records')
summary['section2'] = section2

# Section 3: Target Analysis
print('Section 3: targets...')
section3 = {}

train_scored = train[train['need_prediction'].astype(bool)].copy()
valid_scored = valid[valid['need_prediction'].astype(bool)].copy()

# Target distribution stats
target_stats = []
for split_name, df in [('train', train_scored), ('valid', valid_scored)]:
    for target in TARGET_COLS:
        vals = df[target].to_numpy()
        target_stats.append({
            'split': split_name,
            'target': target,
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'min': float(np.min(vals)),
            'max': float(np.max(vals)),
            'skew': float(pd.Series(vals).skew()),
            'kurtosis': float(pd.Series(vals).kurtosis()),
            'q50_abs': float(np.quantile(np.abs(vals), 0.50)),
            'q90_abs': float(np.quantile(np.abs(vals), 0.90)),
            'q95_abs': float(np.quantile(np.abs(vals), 0.95)),
            'q99_abs': float(np.quantile(np.abs(vals), 0.99)),
        })

pd.DataFrame(target_stats).to_csv(ART_DIR / 'target_summary_stats.csv', index=False)

# Target histograms
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for r, target in enumerate(TARGET_COLS):
    train_vals = train_scored[target].to_numpy()
    valid_vals = valid_scored[target].to_numpy()
    lo = float(np.percentile(np.concatenate([train_vals, valid_vals]), 0.5))
    hi = float(np.percentile(np.concatenate([train_vals, valid_vals]), 99.5))
    bins = np.linspace(lo, hi, 80)

    axes[r, 0].hist(train_vals, bins=bins, density=True, alpha=0.7, color='#1f77b4')
    axes[r, 0].set_title(f'{target} train distribution (scored steps)')

    axes[r, 1].hist(valid_vals, bins=bins, density=True, alpha=0.7, color='#ff7f0e')
    axes[r, 1].set_title(f'{target} valid distribution (scored steps)')

fig.tight_layout()
fig.savefig(ART_DIR / 'target_distributions.png', dpi=160)
plt.close(fig)

# Box plot for targets
fig, ax = plt.subplots(figsize=(10, 5))
box_data = [
    train_scored['t0'].to_numpy(),
    train_scored['t1'].to_numpy(),
    valid_scored['t0'].to_numpy(),
    valid_scored['t1'].to_numpy(),
]
ax.boxplot(box_data, showfliers=False, labels=['train_t0', 'train_t1', 'valid_t0', 'valid_t1'])
ax.set_title('Target Box Plots (scored steps, outliers hidden)')
fig.tight_layout()
fig.savefig(ART_DIR / 'target_boxplots.png', dpi=160)
plt.close(fig)

# t0/t1 correlation
section3['train_t0_t1_corr'] = float(train_scored[['t0', 't1']].corr().iloc[0, 1])
section3['valid_t0_t1_corr'] = float(valid_scored[['t0', 't1']].corr().iloc[0, 1])

# Weight concentration: share of |target| from top X%
def weight_concentration(vals: np.ndarray, quantiles=(0.01, 0.05, 0.1, 0.2)):
    abs_vals = np.abs(vals)
    total = abs_vals.sum()
    rows = {}
    for q in quantiles:
        thr = np.quantile(abs_vals, 1.0 - q)
        share = abs_vals[abs_vals >= thr].sum() / (total + 1e-12)
        rows[f'top_{int(q*100)}pct_share'] = float(share)
        rows[f'top_{int(q*100)}pct_threshold'] = float(thr)
    return rows

section3['weight_concentration'] = {
    'train_t0': weight_concentration(train_scored['t0'].to_numpy()),
    'train_t1': weight_concentration(train_scored['t1'].to_numpy()),
    'valid_t0': weight_concentration(valid_scored['t0'].to_numpy()),
    'valid_t1': weight_concentration(valid_scored['t1'].to_numpy()),
}

# Target autocorrelation at lags
lags = [1, 2, 5, 10, 20, 50, 100]
autocorr_rows = []
n_train_seq = section1['n_train_sequences']

for target in TARGET_COLS:
    arr = train[target].to_numpy().reshape(n_train_seq, 1000)
    for lag in lags:
        x = arr[:, :-lag].ravel()
        y = arr[:, lag:].ravel()
        corr = np.corrcoef(x, y)[0, 1]
        autocorr_rows.append({'target': target, 'lag': lag, 'autocorr': float(corr)})

autocorr_df = pd.DataFrame(autocorr_rows)
autocorr_df.to_csv(ART_DIR / 'target_autocorrelation.csv', index=False)
section3['target_autocorr'] = autocorr_rows

# Large-move clustering at lag 1
cluster = {}
for target in TARGET_COLS:
    arr = np.abs(train[target].to_numpy().reshape(n_train_seq, 1000))
    thr = np.quantile(arr, 0.95)
    large = arr >= thr
    base_rate = large[:, 1:].mean()
    cond_rate = large[:, 1:][large[:, :-1]].mean()
    cluster[target] = {
        'threshold_abs_q95': float(thr),
        'base_rate_next_step': float(base_rate),
        'cond_rate_next_given_current_large': float(cond_rate),
        'lift': float(cond_rate / (base_rate + 1e-12)),
    }
section3['large_move_clustering'] = cluster

summary['section3'] = section3

# Section 4: Feature-target relationships
print('Section 4: feature-target relationships...')
section4 = {}

corr_df = train_scored[FEATURE_COLS + TARGET_COLS].corr()
ft_corr = corr_df.loc[FEATURE_COLS, TARGET_COLS].copy()
ft_corr.to_csv(ART_DIR / 'feature_target_pearson_corr.csv')

rows = []
for target in TARGET_COLS:
    order = ft_corr[target].abs().sort_values(ascending=False)
    top = order.head(5)
    section4[f'top5_{target}_pearson_abs'] = [
        {'feature': f, 'corr': float(ft_corr.loc[f, target])} for f in top.index
    ]

    for f in top.index:
        rho, _ = spearmanr(train_scored[f].to_numpy(), train_scored[target].to_numpy())
        rows.append({'target': target, 'feature': f, 'pearson': float(ft_corr.loc[f, target]), 'spearman': float(rho)})

pd.DataFrame(rows).to_csv(ART_DIR / 'top_feature_pearson_vs_spearman.csv', index=False)

# Scatter plots for top features (sampled)
scatter_n = 200_000
sample_idx = RNG.choice(len(train_scored), size=min(scatter_n, len(train_scored)), replace=False)
sample_df = train_scored.iloc[sample_idx]

for target in TARGET_COLS:
    top_feats = [x['feature'] for x in section4[f'top5_{target}_pearson_abs']]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for i, feat in enumerate(top_feats):
        ax = axes[i]
        ax.scatter(sample_df[feat], sample_df[target], s=2, alpha=0.15, color='#1f77b4')
        ax.set_xlabel(feat)
        ax.set_ylabel(target)
        ax.set_title(f'{feat} vs {target}')
    for j in range(len(top_feats), len(axes)):
        axes[j].axis('off')
    fig.tight_layout()
    fig.savefig(ART_DIR / f'scatter_top5_{target}.png', dpi=160)
    plt.close(fig)

summary['section4'] = section4

# Section 5: Temporal structure
print('Section 5: temporal structure...')
section5 = {}

# Example sequences plot
example_seq_ids = train['seq_ix'].drop_duplicates().sample(3, random_state=42).tolist()
plot_cols = ['p0', 'p6', 'v0', 'v6', 'dp0', 'dv0', 't0', 't1']
fig, axes = plt.subplots(len(plot_cols), len(example_seq_ids), figsize=(18, 20), sharex=True)
for c, seq_id in enumerate(example_seq_ids):
    seq_df = train[train['seq_ix'] == seq_id].sort_values('step_in_seq')
    for r, col in enumerate(plot_cols):
        ax = axes[r, c]
        ax.plot(seq_df['step_in_seq'].to_numpy(), seq_df[col].to_numpy(), lw=0.8)
        if r == 0:
            ax.set_title(f'seq_ix={seq_id}')
        if c == 0:
            ax.set_ylabel(col)
        if r == len(plot_cols)-1:
            ax.set_xlabel('step_in_seq')
fig.tight_layout()
fig.savefig(ART_DIR / 'example_sequences.png', dpi=160)
plt.close(fig)
section5['example_seq_ids'] = [int(x) for x in example_seq_ids]

# Selected feature autocorrelation
selected_features = ['p0', 'p6', 'v0', 'v6', 'dp0', 'dv0']
feat_ac_rows = []
for feat in selected_features:
    arr = train[feat].to_numpy().reshape(n_train_seq, 1000)
    for lag in lags:
        x = arr[:, :-lag].ravel()
        y = arr[:, lag:].ravel()
        corr = np.corrcoef(x, y)[0, 1]
        feat_ac_rows.append({'feature': feat, 'lag': lag, 'autocorr': float(corr)})
feat_ac_df = pd.DataFrame(feat_ac_rows)
feat_ac_df.to_csv(ART_DIR / 'selected_feature_autocorrelation.csv', index=False)
section5['selected_feature_autocorr'] = feat_ac_rows

# Stationarity diagnostics: mean/std by step and drift ratio
stationarity_cols = ['p0', 'p6', 'v0', 'v6', 'dp0', 'dv0', 't0', 't1']
step_stats = train.groupby('step_in_seq')[stationarity_cols].agg(['mean', 'std'])
step_stats.to_csv(ART_DIR / 'stepwise_mean_std.csv')

stationarity_summary = {}
for col in stationarity_cols:
    step_mean = step_stats[(col, 'mean')]
    overall_std = train[col].std()
    drift_ratio = (step_mean.max() - step_mean.min()) / (overall_std + 1e-12)
    stationarity_summary[col] = float(drift_ratio)
section5['drift_ratio_max_min_over_global_std'] = stationarity_summary

# Between-sequence heterogeneity: std of per-sequence means relative to global std
seq_means = train.groupby('seq_ix')[FEATURE_COLS + TARGET_COLS].mean()
hetero = {}
for col in FEATURE_COLS + TARGET_COLS:
    hetero[col] = float(seq_means[col].std() / (train[col].std() + 1e-12))
hetero_df = pd.DataFrame({'feature': list(hetero.keys()), 'seq_mean_std_over_global_std': list(hetero.values())})
hetero_df.to_csv(ART_DIR / 'sequence_heterogeneity.csv', index=False)
section5['top_sequence_heterogeneity'] = hetero_df.sort_values('seq_mean_std_over_global_std', ascending=False).head(10).to_dict(orient='records')

summary['section5'] = section5

# Section 6: Cross-feature patterns
print('Section 6: derived features...')
section6 = {}

derived = pd.DataFrame(index=train_scored.index)
for i in range(6):
    derived[f'spread_{i}'] = train_scored[f'p{6+i}'] - train_scored[f'p{i}']
    b = train_scored[f'v{i}']
    a = train_scored[f'v{6+i}']
    derived[f'imbalance_{i}'] = (b - a) / (b + a + 1e-6)

derived['bid_pressure'] = train_scored[[f'v{i}' for i in range(6)]].sum(axis=1)
derived['ask_pressure'] = train_scored[[f'v{6+i}' for i in range(6)]].sum(axis=1)
derived['pressure_imbalance'] = (derived['bid_pressure'] - derived['ask_pressure']) / (derived['bid_pressure'] + derived['ask_pressure'] + 1e-6)
derived['trade_intensity'] = train_scored[[f'dv{i}' for i in range(4)]].sum(axis=1)

# Correlations with targets
corr_rows = []
for col in derived.columns:
    vals = derived[col].to_numpy()
    for t in TARGET_COLS:
        tvals = train_scored[t].to_numpy()
        pear = np.corrcoef(vals, tvals)[0, 1]
        rho, _ = spearmanr(vals, tvals)
        corr_rows.append({'feature': col, 'target': t, 'pearson': float(pear), 'spearman': float(rho)})

derived_corr = pd.DataFrame(corr_rows)
derived_corr['abs_pearson'] = derived_corr['pearson'].abs()
derived_corr = derived_corr.sort_values(['target', 'abs_pearson'], ascending=[True, False])
derived_corr.to_csv(ART_DIR / 'derived_feature_target_corr.csv', index=False)

section6['top5_derived_t0'] = derived_corr[derived_corr['target'] == 't0'].head(5)[['feature', 'pearson', 'spearman']].to_dict(orient='records')
section6['top5_derived_t1'] = derived_corr[derived_corr['target'] == 't1'].head(5)[['feature', 'pearson', 'spearman']].to_dict(orient='records')

# Distribution plots for main derived signals
main_derived = ['spread_0', 'spread_1', 'imbalance_0', 'imbalance_1', 'pressure_imbalance', 'trade_intensity']
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(main_derived):
    vals = derived[col].to_numpy()
    lo = float(np.percentile(vals, 0.5))
    hi = float(np.percentile(vals, 99.5))
    bins = np.linspace(lo, hi, 80)
    axes[i].hist(vals, bins=bins, density=True, alpha=0.75, color='#2ca02c')
    axes[i].set_title(col)
fig.tight_layout()
fig.savefig(ART_DIR / 'derived_feature_distributions.png', dpi=160)
plt.close(fig)

summary['section6'] = section6

# Key question synthesis
print('Synthesizing key answers...')
key_answers = {}

# 1 normalization
key_answers['normalization'] = {
    'std_ratio_max_over_min': section2['feature_std_max_over_min'],
    'note': 'Large cross-feature scale spread suggests normalization is required; global train-fit z-score is a safe baseline, with optional per-sequence centering ablation for drift-sensitive features.'
}

# 2 feature importance
key_answers['top_features_t0'] = section4['top5_t0_pearson_abs']
key_answers['top_features_t1'] = section4['top5_t1_pearson_abs']

# 3 derived features utility
raw_best_t0 = max(abs(x['corr']) for x in section4['top5_t0_pearson_abs'])
raw_best_t1 = max(abs(x['corr']) for x in section4['top5_t1_pearson_abs'])
der_best_t0 = float(max(abs(x['pearson']) for x in section6['top5_derived_t0']))
der_best_t1 = float(max(abs(x['pearson']) for x in section6['top5_derived_t1']))
key_answers['derived_feature_value'] = {
    'best_abs_corr_raw_t0': raw_best_t0,
    'best_abs_corr_derived_t0': der_best_t0,
    'best_abs_corr_raw_t1': raw_best_t1,
    'best_abs_corr_derived_t1': der_best_t1,
    'note': 'Derived features show incremental signal and are most useful as additive channels rather than replacements.'
}

# 4 t0 vs t1 relation
key_answers['target_relation'] = {
    'train_corr_t0_t1': section3['train_t0_t1_corr'],
    'valid_corr_t0_t1': section3['valid_t0_t1_corr'],
}

# 5 temporal structure in targets
key_answers['target_temporal'] = {
    'autocorr_lag1_t0': next(x['autocorr'] for x in section3['target_autocorr'] if x['target'] == 't0' and x['lag'] == 1),
    'autocorr_lag1_t1': next(x['autocorr'] for x in section3['target_autocorr'] if x['target'] == 't1' and x['lag'] == 1),
    'large_move_clustering': section3['large_move_clustering'],
}

# 6 distribution shift
key_answers['distribution_shift'] = {
    'top_shift_features': section2['largest_shift_features'][:5],
}

# 7 weight concentration
key_answers['weight_concentration'] = section3['weight_concentration']

summary['key_answers'] = key_answers

save_json(summary, ART_DIR / 'eda_summary.json')

print('Done. Artifacts written to', ART_DIR)
