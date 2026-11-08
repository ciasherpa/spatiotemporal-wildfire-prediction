"""
common.py  --  shared settings + data prep used by BOTH models
(rf_baseline.py and convlstm_v2.py), so they are evaluated on exactly
the same months and the same (location, month) rows.

TASK (same for every model): use the previous months to predict whether
there is a fire in the NEXT month at a location.
    - baseline RF : sees only the most recent month (t-1)
    - + temporal  : also sees t-2 and the 3-month rolling average
    - + spatial   : also sees the average of nearby locations
    - ConvLSTM    : sees the last SEQ_LEN months on the whole grid
TEST SET: the last TEST_FRAC of the months (never shuffled).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score)

# ---------------- settings you can play with ----------------
CSV_PATH = '../Model inputs/data/input_FireCCI_samples.csv'
OUTPUT_DIR = '../Model outputs/Comparison/'

FEATURE_VARS = ['pet', 'tmn', 'cld', 'dtr', 'vap', 'wet', 'pre', 'tmp', 'tmx',
                'frs', 'RH', 'SVP', 'VPD', 'windspeed', 'ndvi', 'GPP', 'popd',
                'PDSI', 'SMroot', 'SMsurf']
TARGET_VAR = 'BA_km2'

SEQ_LEN = 3          # how many past months the models are allowed to see
TEST_FRAC = 0.2      # last 20% of months = test set
THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6]   # fire/no-fire cutoffs to try
# ------------------------------------------------------------


def load_data(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    needed = FEATURE_VARS + [TARGET_VAR, 'row', 'column', 'Year', 'Month']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"These columns are missing from the CSV: {missing}")

    df[FEATURE_VARS] = df[FEATURE_VARS].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET_VAR]).copy()
    df['fire'] = (df[TARGET_VAR] > 0).astype(int)

    # t = 0,1,2,... one number per calendar month, in time order
    df['time_key'] = df['Year'] * 100 + df['Month']
    keys = sorted(df['time_key'].unique())
    df['t'] = df['time_key'].map({k: i for i, k in enumerate(keys)})

    # dense grid positions (same mapping the ConvLSTM uses)
    rows = sorted(df['row'].unique())
    cols = sorted(df['column'].unique())
    df['row_idx'] = df['row'].map({r: i for i, r in enumerate(rows)})
    df['col_idx'] = df['column'].map({c: i for i, c in enumerate(cols)})
    df['loc'] = df.groupby(['row_idx', 'col_idx']).ngroup()
    df = df.drop_duplicates(['loc', 't']).reset_index(drop=True)

    n_time = len(keys)
    cutoff_t = int(n_time * (1 - TEST_FRAC))   # targets with t >= cutoff are test
    print(f"{df['loc'].nunique()} locations, {n_time} months, grid {len(rows)}x{len(cols)}")
    print(f"Train targets: months {SEQ_LEN}..{cutoff_t - 1} | Test targets: months {cutoff_t}..{n_time - 1}")
    print(f"Overall fire rate: {df['fire'].mean():.2%}")
    return df, n_time, len(rows), len(cols), cutoff_t


def build_grid(df, n_time, n_rows, n_cols):
    """Flat table -> (time, height, width, features) tensor + labels + 'is this a real cell' mask."""
    X = np.full((n_time, n_rows, n_cols, len(FEATURE_VARS)), np.nan, dtype=np.float32)
    y = np.zeros((n_time, n_rows, n_cols), dtype=np.float32)
    valid = np.zeros((n_time, n_rows, n_cols), dtype=bool)
    t, r, c = df['t'].values, df['row_idx'].values, df['col_idx'].values
    X[t, r, c, :] = df[FEATURE_VARS].values.astype(np.float32)
    y[t, r, c] = df['fire'].values
    valid[t, r, c] = True
    return X, y, valid


def evaluate_thresholds(y_true, probs, thresholds=THRESHOLDS):
    """Precision / recall / F1 at several cutoffs + average precision (cutoff-free)."""
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs)
    ap = average_precision_score(y_true, probs) if y_true.sum() > 0 else np.nan
    out = []
    for th in thresholds:
        pred = (probs >= th).astype(int)
        out.append({
            'threshold': th,
            'precision': precision_score(y_true, pred, zero_division=0),
            'recall': recall_score(y_true, pred, zero_division=0),
            'f1': f1_score(y_true, pred, zero_division=0),
            'avg_precision': ap,
            'n_test_rows': len(y_true),
            'test_fire_rate': y_true.mean(),
        })
    return out
