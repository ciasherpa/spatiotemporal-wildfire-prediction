"""
experiments.py  --  more models + smarter features, evaluated the SAME way as rf_baseline.py
and convlstm_v2.py (same months, same rows, via common.py).

Run:  python experiments.py                (uses CSV_PATH from common.py)
      python experiments.py path/to/data.csv

WHAT IT TESTS
  Reference points (any real model has to beat these):
    - Persistence             : "fire next month" = "fire last month"
    - Climatology (month)     : average fire rate of that calendar month in the training years
    - Climatology (location)  : average fire rate of that location in the training years
  Feature sets (each adds to the one before):
    A  last month's climate/vegetation (what rf_baseline.py's baseline uses)
    B  + time of year (sin/cos of the target month) -- rf_baseline.py never tells the model the season
    C  + fire history (fire 1, 2, 3 and 12 months ago, fires in the last 3 months)
    D  + neighbours (nearby locations' climate AND recent fires, as in rf_baseline.py)
  Models:
    Logistic regression, Random Forest, Extra Trees, HistGradientBoosting,
    plus LightGBM and XGBoost if they are installed.

HOW IT IS SCORED (and why it differs a little from rf_baseline.py)
  - The fire/no-fire cutoff is picked on a VALIDATION slice (the last VAL_FRAC of the training
    months), never on the test set. Picking the "best F1 cutoff" on the test set flatters every model.
  - avg_precision (area under the precision-recall curve) needs no cutoff and is the main number.
  - ap_lo / ap_hi: 95% interval from resampling whole test MONTHS (rows in the same month are
    not independent, so resampling single rows would give intervals that are too narrow).
    If two models' intervals overlap heavily, treat them as tied.
Output: experiments_results.csv (same column names compare_results.py reads)
        experiments_importance.csv (tree-model feature importances for the richest feature set)
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_score, recall_score, f1_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from common import load_data, FEATURE_VARS, SEQ_LEN, OUTPUT_DIR
from rf_baseline import add_lag, add_neighbor_features

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

warnings.filterwarnings('ignore')

# ---------------- settings you can play with ----------------
NEIGHBOR_RADIUS = 2      # same meaning as in rf_baseline.py
FIRE_LAGS = (1, 2, 3, 12)  # months back to look at the fire label itself
VAL_FRAC = 0.15          # last 15% of training months -> used only to pick the cutoff
N_BOOT = 200             # month-resamples for the avg_precision interval
THRESHOLD_GRID = np.arange(0.05, 0.96, 0.05)
SEED = 1
# ------------------------------------------------------------


# ---------------------------------------------------------------- features
def build_features(df):
    feat = df[['loc', 't', 'fire', 'Month']].copy()

    feat = add_lag(feat, df, FEATURE_VARS, 1)               # A: last month's climate
    for k in FIRE_LAGS:                                      # C: fire history
        feat = add_lag(feat, df, ['fire'], k)
    feat['fire_sum3'] = feat[['fire_lag1', 'fire_lag2', 'fire_lag3']].sum(axis=1)

    feat['month_sin'] = np.sin(2 * np.pi * feat['Month'] / 12)   # B: season of the TARGET month
    feat['month_cos'] = np.cos(2 * np.pi * feat['Month'] / 12)   # (known in advance, no leakage)

    # D: neighbours' climate and neighbours' fires, both from the previous month
    feat = add_neighbor_features(feat, df, FEATURE_VARS + ['fire'], NEIGHBOR_RADIUS)
    return feat


def feature_sets():
    A = [f'{v}_lag1' for v in FEATURE_VARS]
    B = A + ['month_sin', 'month_cos']
    C = B + [f'fire_lag{k}' for k in FIRE_LAGS] + ['fire_sum3']
    D = C + [f'{v}_nbr' for v in FEATURE_VARS] + ['fire_nbr']
    return {'A last-month climate': A, 'B +season': B,
            'C +season +fire history': C, 'D +season +fire history +neighbours': D}


# ---------------------------------------------------------------- models
def make_models(pos_rate):
    spw = (1 - pos_rate) / max(pos_rate, 1e-6)   # "a missed fire counts this much more"
    models = {
        'LogReg': make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, class_weight='balanced')),
        'RF': RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                     class_weight='balanced_subsample',
                                     n_jobs=-1, random_state=SEED),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=300, min_samples_leaf=3,
                                           class_weight='balanced_subsample',
                                           n_jobs=-1, random_state=SEED),
        'HistGB': HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                 random_state=SEED),
    }
    if LGBMClassifier is not None:
        models['LightGBM'] = LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                                            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                            scale_pos_weight=spw, random_state=SEED,
                                            n_jobs=-1, verbose=-1)
    if XGBClassifier is not None:
        models['XGBoost'] = XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=5,
                                          subsample=0.8, colsample_bytree=0.8,
                                          scale_pos_weight=spw, random_state=SEED,
                                          n_jobs=-1, eval_metric='logloss')
    return models


def fit_predict(name, model, X_fit, y_fit, X_eval):
    if name == 'HistGB':   # no class_weight option in older scikit-learn, so weight the rows
        model.fit(X_fit, y_fit, sample_weight=compute_sample_weight('balanced', y_fit))
    else:
        model.fit(X_fit, y_fit)
    return model.predict_proba(X_eval)[:, 1]


def baseline_probs(kind, fit, ev):
    overall = fit['fire'].mean()
    if kind == 'persistence':
        return ev['fire_lag1'].fillna(0).values
    if kind == 'clim_month':
        return ev['Month'].map(fit.groupby('Month')['fire'].mean()).fillna(overall).values
    if kind == 'clim_loc':
        return ev['loc'].map(fit.groupby('loc')['fire'].mean()).fillna(overall).values
    raise ValueError(kind)


# ---------------------------------------------------------------- scoring
def choose_threshold(y_val, p_val):
    if y_val.sum() == 0:
        return 0.5
    best_thr, best_f1 = 0.5, -1
    for th in THRESHOLD_GRID:
        f1 = f1_score(y_val, (p_val >= th).astype(int), zero_division=0)
        if f1 > best_f1:
            best_thr, best_f1 = th, f1
    return float(best_thr)


def score(y, p, thr):
    pred = (p >= thr).astype(int)
    return {
        'threshold': round(thr, 2),
        'avg_precision': average_precision_score(y, p) if y.sum() > 0 else np.nan,
        'roc_auc': roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan,
        'precision': precision_score(y, pred, zero_division=0),
        'recall': recall_score(y, pred, zero_division=0),
        'f1': f1_score(y, pred, zero_division=0),
    }


def month_bootstrap_ap(y, p, months, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    uniq = np.unique(months)
    groups = {m: np.where(months == m)[0] for m in uniq}
    aps = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([groups[m] for m in pick])
        if y[idx].sum() > 0:
            aps.append(average_precision_score(y[idx], p[idx]))
    if not aps:
        return np.nan, np.nan
    lo, hi = np.percentile(aps, [2.5, 97.5])
    return lo, hi


def paired_month_bootstrap(y, p_ref, p_new, months, n_boot=N_BOOT, seed=SEED):
    """Resample test months, score BOTH models on the same resample, look at the AP difference.
    Because both see the same months, this is a much sharper comparison than eyeballing two
    separate intervals. Returns (mean diff, 2.5%, 97.5%, share of resamples where new > ref)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(months)
    groups = {m: np.where(months == m)[0] for m in uniq}
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([groups[m] for m in pick])
        if y[idx].sum() > 0:
            diffs.append(average_precision_score(y[idx], p_new[idx])
                         - average_precision_score(y[idx], p_ref[idx]))
    if not diffs:
        return np.nan, np.nan, np.nan, np.nan
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return diffs.mean(), lo, hi, (diffs > 0).mean()


# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df, n_time, n_rows, n_cols, cutoff_t = load_data(sys.argv[1]) if len(sys.argv) > 1 else load_data()

    feat = build_features(df)
    feat = feat[feat['t'] >= SEQ_LEN].reset_index(drop=True)   # same targets as the other scripts
    train = feat[feat['t'] < cutoff_t]
    test = feat[feat['t'] >= cutoff_t]

    val_cut = SEQ_LEN + int((cutoff_t - SEQ_LEN) * (1 - VAL_FRAC))
    fit_part = train[train['t'] < val_cut]
    val_part = train[train['t'] >= val_cut]
    if len(fit_part) == 0 or len(val_part) == 0 or fit_part['fire'].sum() == 0:
        raise ValueError("Train/validation split is empty or has no fires -- check month counts.")

    y_val, y_test = val_part['fire'].values, test['fire'].values
    test_months = test['t'].values
    print(f"\nfit rows {len(fit_part)} | validation rows {len(val_part)} | test rows {len(test)}")
    print(f"fire rate: fit {fit_part['fire'].mean():.2%}, val {y_val.mean():.2%}, test {y_test.mean():.2%}")
    print("Models available:", ', '.join(make_models(0.1).keys()), "\n")

    results = []
    test_probs = {}

    def record(name, n_feat, p_val, p_test):
        test_probs[name] = p_test
        thr = choose_threshold(y_val, p_val)
        row = score(y_test, p_test, thr)
        lo, hi = month_bootstrap_ap(y_test, p_test, test_months)
        row.update(model=name, n_features=n_feat, ap_lo=lo, ap_hi=hi,
                   n_test_rows=len(y_test), test_fire_rate=y_test.mean())
        results.append(row)
        print(f"{name:55s} AP={row['avg_precision']:.3f} [{lo:.3f}-{hi:.3f}]  "
              f"P={row['precision']:.2f} R={row['recall']:.2f} F1={row['f1']:.3f} (cut {row['threshold']})")

    # --- reference points
    for label, kind in [('Persistence (fire last month)', 'persistence'),
                        ('Climatology (calendar month)', 'clim_month'),
                        ('Climatology (location)', 'clim_loc')]:
        record(label, 0, baseline_probs(kind, fit_part, val_part), baseline_probs(kind, train, test))

    # --- models x feature sets
    importance = {}
    sets = feature_sets()
    last_set = list(sets)[-1]
    for fs_name, cols in sets.items():
        med_fit = fit_part[cols].median()
        Xfit = fit_part[cols].fillna(med_fit).fillna(0)
        Xval = val_part[cols].fillna(med_fit).fillna(0)
        med_tr = train[cols].median()
        Xtr = train[cols].fillna(med_tr).fillna(0)
        Xte = test[cols].fillna(med_tr).fillna(0)

        for mname, proto in make_models(train['fire'].mean()).items():
            p_val = fit_predict(mname, clone(proto), Xfit, fit_part['fire'], Xval)   # picks the cutoff
            final = clone(proto)
            p_test = fit_predict(mname, final, Xtr, train['fire'], Xte)             # refit on all training
            record(f'{mname} | {fs_name}', len(cols), p_val, p_test)
            if fs_name == last_set and hasattr(final, 'feature_importances_'):
                imp_raw = np.asarray(final.feature_importances_, dtype=float)
                total = imp_raw.sum()
                # LightGBM reports raw split counts, others report fractions -> put all on one scale
                importance[mname] = pd.Series(imp_raw / total if total > 0 else imp_raw, index=cols)

    out = pd.DataFrame(results)
    path = os.path.join(OUTPUT_DIR, 'experiments_results.csv')
    out.to_csv(path, index=False)
    print(f"\nSaved: {path}")

    # --- paired comparison against the closest thing to the parent paper's approach
    ref_name = f'RF | {list(sets)[0]}'
    if ref_name in test_probs:
        rows = []
        for name, p in test_probs.items():
            if name == ref_name:
                continue
            d, lo, hi, share = paired_month_bootstrap(y_test, test_probs[ref_name], p, test_months)
            rows.append({'model': name, 'ap_gain_vs_baseline': d, 'gain_lo': lo, 'gain_hi': hi,
                         'share_of_resamples_better': share})
        vs = pd.DataFrame(rows).sort_values('ap_gain_vs_baseline', ascending=False)
        vs.to_csv(os.path.join(OUTPUT_DIR, 'experiments_vs_baseline.csv'), index=False)
        print(f"\n=== Paired comparison vs '{ref_name}' (AP gain; interval from resampling test months) ===")
        print("If the interval excludes 0, the gain is unlikely to be luck of the test months.")
        print(vs.round(3).head(12).to_string(index=False))

    if importance:
        imp = pd.DataFrame(importance)
        imp = imp.loc[imp.mean(axis=1).sort_values(ascending=False).index]
        imp.to_csv(os.path.join(OUTPUT_DIR, 'experiments_importance.csv'))
        print("\nTop features (mean importance across tree models, richest feature set):")
        print(imp.mean(axis=1).head(12).round(3).to_string())

    show = ['model', 'avg_precision', 'ap_lo', 'ap_hi', 'roc_auc', 'precision', 'recall', 'f1', 'threshold']
    print("\n=== Ranked by avg_precision (a no-skill model scores ~ the test fire rate, "
          f"{y_test.mean():.3f}) ===")
    print(out.sort_values('avg_precision', ascending=False)[show].round(3).head(15).to_string(index=False))


if __name__ == '__main__':
    main()