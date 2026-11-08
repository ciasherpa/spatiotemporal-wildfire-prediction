"""
rf_baseline.py  --  Random Forest ablation (no deep learning, runs in a minute or two)

Trains 4 versions of the SAME model, each given more context:
    1. RF baseline            : last month's snapshot only (what the parent paper does)
    2. RF + temporal          : + 2 months ago, + 3-month rolling average
    3. RF + spatial           : + average of nearby locations
    4. RF + temporal + spatial
Uses the same months for train/test as convlstm_v2.py (see common.py).
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from common import (load_data, evaluate_thresholds, FEATURE_VARS, SEQ_LEN, OUTPUT_DIR)

# ---------------- settings you can play with ----------------
EXTRA_VARS = FEATURE_VARS     # which variables get lag/neighbor versions (try a shorter list!)
NEIGHBOR_RADIUS = 2           # "nearby" = within this many grid cells (2 ~ the ConvLSTM's view)
N_TREES = 300
# ------------------------------------------------------------


def add_lag(feat, base, vars_, k):
    """Attach each location's values from k months earlier (merge on time, so gaps can't misalign)."""
    lagged = base[['loc', 't'] + vars_].copy()
    lagged['t'] = lagged['t'] + k
    lagged = lagged.rename(columns={v: f'{v}_lag{k}' for v in vars_})
    return feat.merge(lagged, on=['loc', 't'], how='left')


def add_neighbor_features(feat, base, vars_, radius):
    """Average of the OTHER locations within `radius` grid cells, from the previous month."""
    locs = base[['loc', 'row_idx', 'col_idx']].drop_duplicates('loc')
    n_loc = int(locs['loc'].max()) + 1
    coords = np.zeros((n_loc, 2), dtype=int)
    coords[locs['loc'].values] = locs[['row_idx', 'col_idx']].values

    T = int(base['t'].max()) + 1
    arr = np.full((T, n_loc, len(vars_)), np.nan)
    arr[base['t'].values, base['loc'].values, :] = base[vars_].values

    nm = np.full_like(arr, np.nan)
    counts = []
    for i in range(n_loc):
        dist = np.maximum(np.abs(coords[:, 0] - coords[i, 0]), np.abs(coords[:, 1] - coords[i, 1]))
        nbrs = np.where((dist <= radius) & (np.arange(n_loc) != i))[0]
        counts.append(len(nbrs))
        if len(nbrs):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                nm[:, i, :] = np.nanmean(arr[:, nbrs, :], axis=1)
    counts = np.array(counts)
    print(f"Neighbors within radius {radius}: avg {counts.mean():.1f} per location, "
          f"{(counts == 0).sum()} locations have none (their spatial features stay empty)")

    tt, ll = np.meshgrid(np.arange(T), np.arange(n_loc), indexing='ij')
    out = pd.DataFrame({'t': tt.ravel() + 1, 'loc': ll.ravel()})   # +1: month s predicts month s+1
    for k, v in enumerate(vars_):
        out[f'{v}_nbr'] = nm[:, :, k].ravel()
    return feat.merge(out, on=['loc', 't'], how='left')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df, n_time, n_rows, n_cols, cutoff_t = load_data()

    feat = df[['loc', 't', 'fire']].copy()
    feat = add_lag(feat, df, FEATURE_VARS, 1)                # base: last month
    feat = add_lag(feat, df, EXTRA_VARS, 2)
    feat = add_lag(feat, df, EXTRA_VARS, 3)
    for v in EXTRA_VARS:                                     # 3-month rolling average
        feat[f'{v}_roll3'] = feat[[f'{v}_lag1', f'{v}_lag2', f'{v}_lag3']].mean(axis=1)
    feat = add_neighbor_features(feat, df, EXTRA_VARS, NEIGHBOR_RADIUS)

    base_cols = [f'{v}_lag1' for v in FEATURE_VARS]
    temporal_cols = [f'{v}_lag2' for v in EXTRA_VARS] + [f'{v}_roll3' for v in EXTRA_VARS]
    spatial_cols = [f'{v}_nbr' for v in EXTRA_VARS]

    feat = feat[feat['t'] >= SEQ_LEN]                        # same targets as the ConvLSTM
    train = feat[feat['t'] < cutoff_t]
    test = feat[feat['t'] >= cutoff_t]
    print(f"Train rows: {len(train)} (fire rate {train['fire'].mean():.2%}) | "
          f"Test rows: {len(test)} (fire rate {test['fire'].mean():.2%})")

    variants = {
        'RF baseline (last month only)': base_cols,
        'RF + temporal': base_cols + temporal_cols,
        'RF + spatial': base_cols + spatial_cols,
        'RF + temporal + spatial': base_cols + temporal_cols + spatial_cols,
    }

    results = []
    for name, cols in variants.items():
        medians = train[cols].median()                       # fill gaps using TRAIN data only
        Xtr = train[cols].fillna(medians).fillna(0)
        Xte = test[cols].fillna(medians).fillna(0)
        rf = RandomForestClassifier(n_estimators=N_TREES, min_samples_leaf=3,
                                    class_weight='balanced_subsample',
                                    n_jobs=-1, random_state=1)
        rf.fit(Xtr, train['fire'])
        probs = rf.predict_proba(Xte)[:, 1]
        for row in evaluate_thresholds(test['fire'].values, probs):
            row['model'] = name
            row['n_features'] = len(cols)
            results.append(row)
        best = max((r for r in results if r['model'] == name), key=lambda r: r['f1'])
        print(f"{name:35s} features={len(cols):3d}  avg_precision={best['avg_precision']:.3f}  "
              f"best F1={best['f1']:.3f} (threshold {best['threshold']})")

    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, 'rf_results.csv'), index=False)
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'rf_results.csv')}")


if __name__ == '__main__':
    main()
