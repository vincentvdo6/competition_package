import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.feature_selection import mutual_info_regression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocessing import DerivedFeatureBuilder
from utils import weighted_pearson_correlation

ART_DIR = ROOT / 'notebooks' / 'artifacts' / '02_feature_analysis'
ART_DIR.mkdir(parents=True, exist_ok=True)

RAW_COLS = [f'p{i}' for i in range(12)] + [f'v{i}' for i in range(12)] + [f'dp{i}' for i in range(4)] + [f'dv{i}' for i in range(4)]
DERIVED_COLS = DerivedFeatureBuilder.DERIVED_COLS
TARGET_COLS = ['t0', 't1']

RNG = np.random.default_rng(42)


def wpearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(weighted_pearson_correlation(y_true.astype(np.float64), np.clip(y_pred.astype(np.float64), -6, 6)))


def score_multi(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    t0 = wpearson(y_true[:, 0], y_pred[:, 0])
    t1 = wpearson(y_true[:, 1], y_pred[:, 1])
    return {'t0': t0, 't1': t1, 'avg': (t0 + t1) / 2.0}


def weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    w = w.astype(np.float64)
    w_sum = w.sum() + 1e-12
    mx = (w * x).sum() / w_sum
    my = (w * y).sum() / w_sum
    cov = (w * (x - mx) * (y - my)).sum() / w_sum
    vx = (w * (x - mx) ** 2).sum() / w_sum
    vy = (w * (y - my) ** 2).sum() / w_sum
    return float(cov / (np.sqrt(vx * vy) + 1e-12))


def sample_df(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed).copy()


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    raw = df[RAW_COLS].to_numpy(dtype=np.float32, copy=False)
    derived = DerivedFeatureBuilder.compute(raw)
    derived_df = pd.DataFrame(derived, columns=DERIVED_COLS, index=df.index)
    return pd.concat([df, derived_df], axis=1)


print('Loading train/valid parquet...')
train = pd.read_parquet(ROOT / 'datasets' / 'train.parquet')
valid = pd.read_parquet(ROOT / 'datasets' / 'valid.parquet')

train = train.sort_values(['seq_ix', 'step_in_seq']).reset_index(drop=True)
valid = valid.sort_values(['seq_ix', 'step_in_seq']).reset_index(drop=True)

train_scored = train[train['need_prediction'].astype(bool)].copy()
valid_scored = valid[valid['need_prediction'].astype(bool)].copy()

print('Rows (scored) train:', len(train_scored), 'valid:', len(valid_scored))

# Use large but manageable samples for correlation analysis
train_s = sample_df(train_scored, n=700_000, seed=42)
valid_s = sample_df(valid_scored, n=260_000, seed=43)
train_s = add_derived(train_s)
valid_s = add_derived(valid_s)

summary = {
    'data': {
        'train_rows_scored': int(len(train_scored)),
        'valid_rows_scored': int(len(valid_scored)),
        'train_sample_rows': int(len(train_s)),
        'valid_sample_rows': int(len(valid_s)),
    }
}

# ---------------------------------------------------------------------
# 1) Contribution of existing 10 derived features
# ---------------------------------------------------------------------
print('Analyzing derived feature contributions...')
rows = []
for col in DERIVED_COLS:
    tr_x = train_s[col].to_numpy()
    va_x = valid_s[col].to_numpy()

    tr_t0 = train_s['t0'].to_numpy()
    tr_t1 = train_s['t1'].to_numpy()
    va_t0 = valid_s['t0'].to_numpy()
    va_t1 = valid_s['t1'].to_numpy()

    tr_w0 = np.abs(tr_t0)
    tr_w1 = np.abs(tr_t1)
    va_w0 = np.abs(va_t0)
    va_w1 = np.abs(va_t1)

    rows.append({
        'feature': col,
        'train_pearson_t0': float(np.corrcoef(tr_x, tr_t0)[0, 1]),
        'train_pearson_t1': float(np.corrcoef(tr_x, tr_t1)[0, 1]),
        'valid_pearson_t0': float(np.corrcoef(va_x, va_t0)[0, 1]),
        'valid_pearson_t1': float(np.corrcoef(va_x, va_t1)[0, 1]),
        'train_spearman_t0': float(spearmanr(tr_x, tr_t0).correlation),
        'train_spearman_t1': float(spearmanr(tr_x, tr_t1).correlation),
        'valid_spearman_t0': float(spearmanr(va_x, va_t0).correlation),
        'valid_spearman_t1': float(spearmanr(va_x, va_t1).correlation),
        'train_weightedcorr_t0': weighted_corr(tr_x, tr_t0, tr_w0),
        'train_weightedcorr_t1': weighted_corr(tr_x, tr_t1, tr_w1),
        'valid_weightedcorr_t0': weighted_corr(va_x, va_t0, va_w0),
        'valid_weightedcorr_t1': weighted_corr(va_x, va_t1, va_w1),
    })

derived_corr = pd.DataFrame(rows)
derived_corr['valid_abs_sum'] = derived_corr['valid_pearson_t0'].abs() + derived_corr['valid_pearson_t1'].abs()
derived_corr = derived_corr.sort_values('valid_abs_sum', ascending=False)
derived_corr.to_csv(ART_DIR / 'derived_feature_contribution_rank.csv', index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_df = derived_corr.sort_values('valid_pearson_t0')
axes[0].barh(plot_df['feature'], plot_df['valid_pearson_t0'], color='#1f77b4')
axes[0].set_title('Derived Feature Corr with t0 (valid sample)')
axes[0].set_xlabel('Pearson corr')
plot_df2 = derived_corr.sort_values('valid_pearson_t1')
axes[1].barh(plot_df2['feature'], plot_df2['valid_pearson_t1'], color='#ff7f0e')
axes[1].set_title('Derived Feature Corr with t1 (valid sample)')
axes[1].set_xlabel('Pearson corr')
plt.tight_layout()
plt.savefig(ART_DIR / 'derived_feature_corr_bars.png', dpi=170)
plt.close(fig)

# Permutation importance with lightweight linear proxy model on derived-only features
print('Running derived-feature permutation importance...')
train_r = sample_df(train_s, n=450_000, seed=44)
valid_r = sample_df(valid_s, n=180_000, seed=45)

X_train_d = train_r[DERIVED_COLS].to_numpy(dtype=np.float32)
Y_train = train_r[TARGET_COLS].to_numpy(dtype=np.float32)
X_valid_d = valid_r[DERIVED_COLS].to_numpy(dtype=np.float32)
Y_valid = valid_r[TARGET_COLS].to_numpy(dtype=np.float32)

ridge_d = make_pipeline(StandardScaler(), Ridge(alpha=2.0))
ridge_d.fit(X_train_d, Y_train)
pred_base = ridge_d.predict(X_valid_d)
base_scores = score_multi(Y_valid, pred_base)

perm_rows = []
for j, col in enumerate(DERIVED_COLS):
    Xp = X_valid_d.copy()
    Xp[:, j] = Xp[RNG.permutation(len(Xp)), j]
    pred_p = ridge_d.predict(Xp)
    sc = score_multi(Y_valid, pred_p)
    perm_rows.append({
        'feature': col,
        'base_avg': base_scores['avg'],
        'perm_avg': sc['avg'],
        'delta_avg': base_scores['avg'] - sc['avg'],
        'base_t0': base_scores['t0'],
        'perm_t0': sc['t0'],
        'delta_t0': base_scores['t0'] - sc['t0'],
        'base_t1': base_scores['t1'],
        'perm_t1': sc['t1'],
        'delta_t1': base_scores['t1'] - sc['t1'],
    })

perm_df = pd.DataFrame(perm_rows).sort_values('delta_avg', ascending=False)
perm_df.to_csv(ART_DIR / 'derived_feature_permutation_importance.csv', index=False)

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(perm_df['feature'], perm_df['delta_avg'], color='#2ca02c')
ax.set_title('Derived Feature Permutation Importance (delta weighted-pearson avg)')
ax.set_xlabel('Score drop when permuted')
plt.tight_layout()
plt.savefig(ART_DIR / 'derived_feature_permutation_importance.png', dpi=170)
plt.close(fig)

# Derived vs raw vs combined proxy check
X_train_raw = train_r[RAW_COLS].to_numpy(dtype=np.float32)
X_valid_raw = valid_r[RAW_COLS].to_numpy(dtype=np.float32)
X_train_all = train_r[RAW_COLS + DERIVED_COLS].to_numpy(dtype=np.float32)
X_valid_all = valid_r[RAW_COLS + DERIVED_COLS].to_numpy(dtype=np.float32)

ridge_raw = make_pipeline(StandardScaler(), Ridge(alpha=2.0))
ridge_all = make_pipeline(StandardScaler(), Ridge(alpha=2.0))
ridge_raw.fit(X_train_raw, Y_train)
ridge_all.fit(X_train_all, Y_train)

score_raw = score_multi(Y_valid, ridge_raw.predict(X_valid_raw))
score_derived = base_scores
score_all = score_multi(Y_valid, ridge_all.predict(X_valid_all))

model_proxy_scores = {
    'raw_only': score_raw,
    'derived_only': score_derived,
    'raw_plus_derived': score_all,
}

with open(ART_DIR / 'feature_set_proxy_scores.json', 'w', encoding='utf-8') as f:
    json.dump(model_proxy_scores, f, indent=2)

summary['derived_feature_proxy_scores'] = model_proxy_scores

# ---------------------------------------------------------------------
# 2) Deep t1 predictability analysis
# ---------------------------------------------------------------------
print('Running deep t1 predictability analysis...')
full_cols = RAW_COLS + DERIVED_COLS

corr_rows = []
for col in full_cols:
    tx = train_s[col].to_numpy()
    vx = valid_s[col].to_numpy()
    tr_t1 = train_s['t1'].to_numpy()
    va_t1 = valid_s['t1'].to_numpy()
    corr_rows.append({
        'feature': col,
        'train_pearson_t1': float(np.corrcoef(tx, tr_t1)[0, 1]),
        'valid_pearson_t1': float(np.corrcoef(vx, va_t1)[0, 1]),
        'train_spearman_t1': float(spearmanr(tx, tr_t1).correlation),
        'valid_spearman_t1': float(spearmanr(vx, va_t1).correlation),
        'train_weightedcorr_t1': weighted_corr(tx, tr_t1, np.abs(tr_t1)),
        'valid_weightedcorr_t1': weighted_corr(vx, va_t1, np.abs(va_t1)),
    })

t1_corr = pd.DataFrame(corr_rows)
t1_corr['valid_abs_pearson'] = t1_corr['valid_pearson_t1'].abs()
t1_corr = t1_corr.sort_values('valid_abs_pearson', ascending=False)
t1_corr.to_csv(ART_DIR / 't1_feature_correlations_42.csv', index=False)

fig, ax = plt.subplots(figsize=(10, 7))
top_t1 = t1_corr.head(15).sort_values('valid_pearson_t1')
ax.barh(top_t1['feature'], top_t1['valid_pearson_t1'], color='#9467bd')
ax.set_title('Top Features by |corr| with t1 (valid sample)')
ax.set_xlabel('Pearson corr with t1')
plt.tight_layout()
plt.savefig(ART_DIR / 't1_top_feature_corr.png', dpi=170)
plt.close(fig)

# Mutual information (nonlinear signal proxy)
mi_df = sample_df(train_s[full_cols + ['t1']], n=160_000, seed=46)
mi_vals = mutual_info_regression(
    mi_df[full_cols].to_numpy(dtype=np.float32),
    mi_df['t1'].to_numpy(dtype=np.float32),
    random_state=42,
)
mi_out = pd.DataFrame({'feature': full_cols, 'mi_t1': mi_vals}).sort_values('mi_t1', ascending=False)
mi_out.to_csv(ART_DIR / 't1_mutual_information.csv', index=False)

# Cross-feature interaction scan (pair products among top 10 features)
selected = t1_corr.head(10)['feature'].tolist()
int_rows = []
base = train_s[selected + ['t1']].copy()
for a_i in range(len(selected)):
    for b_i in range(a_i + 1, len(selected)):
        a = selected[a_i]
        b = selected[b_i]
        prod = base[a].to_numpy() * base[b].to_numpy()
        corr = float(np.corrcoef(prod, base['t1'].to_numpy())[0, 1])
        int_rows.append({'feature_a': a, 'feature_b': b, 'interaction': f'{a}*{b}', 'pearson_t1': corr, 'abs_corr': abs(corr)})

interactions = pd.DataFrame(int_rows).sort_values('abs_corr', ascending=False)
interactions.to_csv(ART_DIR / 't1_interaction_scan_top10.csv', index=False)

fig, ax = plt.subplots(figsize=(10, 7))
best_int = interactions.head(15).sort_values('pearson_t1')
ax.barh(best_int['interaction'], best_int['pearson_t1'], color='#8c564b')
ax.set_title('Top Interaction Terms by |corr| with t1 (train sample)')
ax.set_xlabel('Pearson corr')
plt.tight_layout()
plt.savefig(ART_DIR / 't1_top_interactions.png', dpi=170)
plt.close(fig)

# Lag analysis: corr(feature_{t-lag}, t1_t)
seq_ids = train['seq_ix'].drop_duplicates().sample(n=min(1800, train['seq_ix'].nunique()), random_state=47)
lag_df = train[train['seq_ix'].isin(seq_ids)][['seq_ix', 'step_in_seq'] + RAW_COLS + ['t1']].copy()
lag_df = lag_df.sort_values(['seq_ix', 'step_in_seq']).reset_index(drop=True)

lag_df = add_derived(lag_df)
lag_features = ['t1'] + t1_corr.head(8)['feature'].tolist()
lag_features = list(dict.fromkeys(lag_features))

n_seq = lag_df['seq_ix'].nunique()
lags = [1, 2, 5, 10, 20, 50]
lag_rows = []
for feat in lag_features:
    arr = lag_df[feat].to_numpy().reshape(n_seq, 1000)
    targ = lag_df['t1'].to_numpy().reshape(n_seq, 1000)
    for lag in lags:
        x = arr[:, :-lag].ravel()
        y = targ[:, lag:].ravel()
        lag_rows.append({
            'feature': feat,
            'lag': lag,
            'corr_feature_t_minus_lag__t1_t': float(np.corrcoef(x, y)[0, 1]),
        })

lag_corr_df = pd.DataFrame(lag_rows)
lag_corr_df.to_csv(ART_DIR / 't1_lag_feature_correlations.csv', index=False)

# ---------------------------------------------------------------------
# 3) Candidate new derived features (rolling/roc/volatility)
# ---------------------------------------------------------------------
print('Evaluating candidate next-round derived features...')


def build_candidate_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[['seq_ix', 'step_in_seq', 'need_prediction', 't1']].copy()
    out['spread_0'] = df['p6'] - df['p0']
    out['spread_2'] = df['p8'] - df['p2']
    out['trade_intensity'] = df[['dv0', 'dv1', 'dv2', 'dv3']].sum(axis=1)
    out['bid_pressure'] = df[[f'v{i}' for i in range(6)]].sum(axis=1)
    out['ask_pressure'] = df[[f'v{i}' for i in range(6, 12)]].sum(axis=1)
    out['pressure_imbalance'] = (out['bid_pressure'] - out['ask_pressure']) / (out['bid_pressure'] + out['ask_pressure'] + 1e-8)
    out['microprice_proxy'] = (df['p0'] * df['v6'] + df['p6'] * df['v0']) / (df['v0'] + df['v6'] + 1e-8)
    out['mid_l1'] = (df['p0'] + df['p6']) / 2.0

    g = out.groupby('seq_ix', sort=False)

    out['spread0_roc1'] = g['spread_0'].diff(1)
    out['spread0_roc5'] = g['spread_0'].diff(5)
    out['pressure_imbalance_roc1'] = g['pressure_imbalance'].diff(1)
    out['trade_intensity_roll_mean_5'] = g['trade_intensity'].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    out['trade_intensity_roll_std_20'] = g['trade_intensity'].rolling(20, min_periods=5).std().reset_index(level=0, drop=True)
    out['dp0_roc1'] = df.groupby('seq_ix', sort=False)['dp0'].diff(1)
    out['dp0_vol_20'] = df.groupby('seq_ix', sort=False)['dp0'].rolling(20, min_periods=5).std().reset_index(level=0, drop=True)
    out['microprice_dev'] = out['microprice_proxy'] - out['mid_l1']
    out['spread2_vol_20'] = g['spread_2'].rolling(20, min_periods=5).std().reset_index(level=0, drop=True)

    # Fill startup NaNs from rolling/diff
    for c in out.columns:
        if c not in ('seq_ix', 'step_in_seq', 'need_prediction', 't1'):
            out[c] = out[c].fillna(0.0)
    return out


train_ids = train['seq_ix'].drop_duplicates().sample(n=min(1700, train['seq_ix'].nunique()), random_state=48)
valid_ids = valid['seq_ix'].drop_duplicates().sample(n=min(500, valid['seq_ix'].nunique()), random_state=49)

cand_train = train[train['seq_ix'].isin(train_ids)].copy()
cand_valid = valid[valid['seq_ix'].isin(valid_ids)].copy()

cand_train = cand_train.sort_values(['seq_ix', 'step_in_seq']).reset_index(drop=True)
cand_valid = cand_valid.sort_values(['seq_ix', 'step_in_seq']).reset_index(drop=True)

cand_train = build_candidate_features(cand_train)
cand_valid = build_candidate_features(cand_valid)

cand_train = cand_train[cand_train['need_prediction'].astype(bool)]
cand_valid = cand_valid[cand_valid['need_prediction'].astype(bool)]

cand_feature_cols = [
    'spread0_roc1',
    'spread0_roc5',
    'pressure_imbalance_roc1',
    'trade_intensity_roll_mean_5',
    'trade_intensity_roll_std_20',
    'dp0_roc1',
    'dp0_vol_20',
    'microprice_dev',
    'spread2_vol_20',
]

cand_rows = []
for col in cand_feature_cols:
    tr_x = cand_train[col].to_numpy()
    va_x = cand_valid[col].to_numpy()
    tr_y = cand_train['t1'].to_numpy()
    va_y = cand_valid['t1'].to_numpy()
    cand_rows.append({
        'feature': col,
        'train_pearson_t1': float(np.corrcoef(tr_x, tr_y)[0, 1]),
        'valid_pearson_t1': float(np.corrcoef(va_x, va_y)[0, 1]),
        'train_spearman_t1': float(spearmanr(tr_x, tr_y).correlation),
        'valid_spearman_t1': float(spearmanr(va_x, va_y).correlation),
    })

cand_corr = pd.DataFrame(cand_rows)
cand_corr['valid_abs_pearson'] = cand_corr['valid_pearson_t1'].abs()
cand_corr = cand_corr.sort_values('valid_abs_pearson', ascending=False)
cand_corr.to_csv(ART_DIR / 'candidate_new_features_t1_corr.csv', index=False)

fig, ax = plt.subplots(figsize=(9, 5))
plot_c = cand_corr.sort_values('valid_pearson_t1')
ax.barh(plot_c['feature'], plot_c['valid_pearson_t1'], color='#17becf')
ax.set_title('Candidate New Derived Features vs t1 (valid subset)')
ax.set_xlabel('Pearson corr with t1')
plt.tight_layout()
plt.savefig(ART_DIR / 'candidate_new_features_t1_corr.png', dpi=170)
plt.close(fig)

# ---------------------------------------------------------------------
# Summary object
# ---------------------------------------------------------------------
summary['derived_feature_top_by_valid_abs'] = (
    derived_corr[['feature', 'valid_pearson_t0', 'valid_pearson_t1', 'valid_weightedcorr_t0', 'valid_weightedcorr_t1']]
    .head(5)
    .to_dict(orient='records')
)
summary['derived_permutation_top'] = perm_df[['feature', 'delta_avg', 'delta_t0', 'delta_t1']].head(5).to_dict(orient='records')
summary['t1_top_features'] = t1_corr[['feature', 'valid_pearson_t1', 'valid_weightedcorr_t1']].head(10).to_dict(orient='records')
summary['t1_top_interactions'] = interactions[['interaction', 'pearson_t1', 'abs_corr']].head(10).to_dict(orient='records')
summary['candidate_new_features_rank'] = cand_corr[['feature', 'valid_pearson_t1', 'valid_spearman_t1']].head(10).to_dict(orient='records')
summary['notes'] = {
    'lag_table_file': 't1_lag_feature_correlations.csv',
    'all_outputs_dir': str(ART_DIR),
}

with open(ART_DIR / 'feature_analysis_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

print('Feature analysis complete. Artifacts saved to', ART_DIR)
