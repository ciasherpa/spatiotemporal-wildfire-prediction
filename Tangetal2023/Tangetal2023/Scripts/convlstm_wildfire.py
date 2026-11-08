#! /bin/env python
# -*- coding: utf-8 -*-
"""
ConvLSTM Wildfire Prediction Pipeline
--------------------------------------
Extends the Tang et al. (2023) TSECfire approach by adding spatial AND
temporal context: instead of predicting fire risk from a single month's
snapshot at a single pixel, this model looks at a SEQUENCE of past months
across a GRID of nearby pixels, and learns spatial + temporal patterns
jointly using a ConvLSTM network.

IMPORTANT — read before running:
This pipeline assumes your data has multiple (Year, Month) records for the
SAME (row, column) location, so there's an actual timeline to learn from.
Before running this script, confirm that with:

    import pandas as pd
    df = pd.read_csv('../Model inputs/data/input_FireCCI_samples.csv')
    print(df.groupby(['row','column']).size().describe())

If the mean there is close to 1, most locations only have a single
snapshot, and this script will run but will not have meaningful sequences
to learn from — the feature-engineering (lag/neighbor) approach discussed
earlier would be the better fit instead.

Install requirements first:
    pip install tensorflow scikit-learn pandas numpy openpyxl --break-system-packages
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, r2_score, mean_absolute_error,
    precision_score, recall_score, f1_score
)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ============================================================
# CONFIG — adjust these for your data / experiment
# ============================================================

CSV_PATH = '../Model inputs/data/input_FireCCI_samples.csv'
OUTPUT_DIR = '../Model outputs/ConvLSTM/'

# Climate/driver variables to use as input channels (grid features).
# These should match columns that exist in your CSV.
FEATURE_VARS = [
    'pet', 'tmn', 'cld', 'dtr', 'vap', 'wet', 'pre', 'tmp', 'tmx',
    'frs', 'RH', 'SVP', 'VPD', 'windspeed', 'ndvi', 'GPP', 'popd',
    'PDSI', 'SMroot', 'SMsurf'
]

# Target variable: burned area. We'll threshold it to a binary
# fire/no-fire target for classification, matching the baseline paper.
TARGET_VAR = 'BA_km2'

# Number of past months to feed the model as one input sequence
# (e.g. 3 means: use months t-2, t-1, t to predict month t+1)
SEQ_LEN = 3

# Train/test split is done BY TIME (earlier years = train, later = test)
# rather than randomly, since shuffling would leak future info into training.
TRAIN_YEAR_CUTOFF = None  # set to e.g. 2015 to split manually; None = auto 80/20 by year

BATCH_SIZE = 8
EPOCHS = 30
RANDOM_SEED = 1

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# STEP 1: Load data and build a (row, column) grid index
# ============================================================

def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)

    missing = [c for c in FEATURE_VARS + [TARGET_VAR, 'row', 'column', 'Year', 'Month'] if c not in df.columns]
    if missing:
        raise ValueError(f"These expected columns are missing from the CSV: {missing}")

    # Binary fire/no-fire target, matching the classification step in the
    # baseline pipeline (target[target > 0] = 1)
    df['fire'] = (df[TARGET_VAR] > 0).astype(int)

    # Map raw row/column values to dense 0-based grid indices, in case the
    # original values are sparse or non-contiguous (e.g. 103, 107, 250...)
    unique_rows = sorted(df['row'].unique())
    unique_cols = sorted(df['column'].unique())
    row_to_idx = {r: i for i, r in enumerate(unique_rows)}
    col_to_idx = {c: i for i, c in enumerate(unique_cols)}

    df['row_idx'] = df['row'].map(row_to_idx)
    df['col_idx'] = df['column'].map(col_to_idx)

    n_rows = len(unique_rows)
    n_cols = len(unique_cols)

    print(f"Grid size: {n_rows} x {n_cols}")
    print(f"Date range: {df['Year'].min()}-{df['Month'].min()} to {df['Year'].max()}-{df['Month'].max()}")

    return df, n_rows, n_cols


# ============================================================
# STEP 2: Turn the flat table into a (time, height, width, channels) tensor
# ============================================================

def build_grid_tensor(df, n_rows, n_cols):
    """
    Returns:
        X_grid: array of shape (n_timesteps, n_rows, n_cols, n_features)
        y_grid: array of shape (n_timesteps, n_rows, n_cols) — binary fire label
        timeline: list of (Year, Month) tuples, in order, matching axis 0
    """
    df = df.copy()
    df['time_key'] = df['Year'] * 100 + df['Month']
    timeline = sorted(df['time_key'].unique())
    time_to_idx = {t: i for i, t in enumerate(timeline)}

    n_time = len(timeline)
    n_features = len(FEATURE_VARS)

    X_grid = np.full((n_time, n_rows, n_cols, n_features), np.nan, dtype=np.float32)
    y_grid = np.full((n_time, n_rows, n_cols), np.nan, dtype=np.float32)

    for _, record in df.iterrows():
        t = time_to_idx[record['time_key']]
        r = int(record['row_idx'])
        c = int(record['col_idx'])
        X_grid[t, r, c, :] = record[FEATURE_VARS].values.astype(np.float32)
        y_grid[t, r, c] = record['fire']

    # Fill missing grid cells (locations with no data at that timestep)
    # with 0 for features and 0 (no fire) for target, since a true NaN
    # can't be fed into ConvLSTM directly. This is a simplifying choice —
    # consider spatial interpolation if too much of the grid is sparse.
    missing_frac = np.isnan(X_grid).mean()
    print(f"Fraction of grid cells with no data at a given timestep: {missing_frac:.2%}")

    X_grid = np.nan_to_num(X_grid, nan=0.0)
    y_grid = np.nan_to_num(y_grid, nan=0.0)

    return X_grid, y_grid, timeline


# ============================================================
# STEP 3: Build (sequence -> next-step) training examples
# ============================================================

def build_sequences(X_grid, y_grid, seq_len):
    """
    Slides a window of length seq_len over the time axis.
    Input:  X_grid[t : t+seq_len]       -> shape (seq_len, H, W, C)
    Target: y_grid[t+seq_len]           -> shape (H, W)
    """
    n_time = X_grid.shape[0]
    X_seqs, y_targets, target_time_idx = [], [], []

    for t in range(n_time - seq_len):
        X_seqs.append(X_grid[t:t + seq_len])
        y_targets.append(y_grid[t + seq_len])
        target_time_idx.append(t + seq_len)

    X_seqs = np.stack(X_seqs)       # (n_samples, seq_len, H, W, C)
    y_targets = np.stack(y_targets)  # (n_samples, H, W)

    return X_seqs, y_targets, target_time_idx


def normalize_features(X_train, X_test):
    """Standardize each feature channel using train-set statistics only."""
    n_features = X_train.shape[-1]
    means = np.zeros(n_features)
    stds = np.zeros(n_features)

    for f in range(n_features):
        vals = X_train[..., f]
        means[f] = vals.mean()
        stds[f] = vals.std() if vals.std() > 1e-8 else 1.0

    X_train_norm = (X_train - means) / stds
    X_test_norm = (X_test - means) / stds
    return X_train_norm, X_test_norm


# ============================================================
# STEP 4: Build the ConvLSTM model
# ============================================================

def build_convlstm_model(seq_len, n_rows, n_cols, n_features):
    model = models.Sequential([
        layers.Input(shape=(seq_len, n_rows, n_cols, n_features)),

        layers.ConvLSTM2D(
            filters=16, kernel_size=(3, 3),
            padding='same', return_sequences=True, activation='tanh'
        ),
        layers.BatchNormalization(),

        layers.ConvLSTM2D(
            filters=8, kernel_size=(3, 3),
            padding='same', return_sequences=False, activation='tanh'
        ),
        layers.BatchNormalization(),

        # Final 1x1 conv collapses feature maps to a single fire-probability
        # value per grid cell
        layers.Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# ============================================================
# STEP 5: Evaluation (same style of metrics as the baseline pipeline)
# ============================================================

def evaluate(y_true, y_pred_prob, threshold=0.5):
    y_true_flat = y_true.flatten()
    y_pred_flat = (y_pred_prob.flatten() > threshold).astype(int)

    acc = accuracy_score(y_true_flat, y_pred_flat)
    precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)

    return {
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    df, n_rows, n_cols = load_and_prepare(CSV_PATH)

    print("Building grid tensor (this can take a while for large grids)...")
    X_grid, y_grid, timeline = build_grid_tensor(df, n_rows, n_cols)

    print("Building sequences...")
    X_seqs, y_targets, target_time_idx = build_sequences(X_grid, y_grid, SEQ_LEN)
    print(f"Total sequences available: {X_seqs.shape[0]}")

    if X_seqs.shape[0] < 10:
        raise ValueError(
            "Fewer than 10 sequences were built. This almost certainly means "
            "your data does not have enough distinct months per location for "
            "ConvLSTM to be meaningful. Re-check df.groupby(['row','column'])."
            "size() before proceeding further."
        )

    # --- Temporal train/test split (NOT random — later time = test) ---
    if TRAIN_YEAR_CUTOFF is not None:
        split_point = next(
            i for i, t in enumerate(target_time_idx)
            if timeline[t] // 100 >= TRAIN_YEAR_CUTOFF
        )
    else:
        split_point = int(len(target_time_idx) * 0.8)

    X_train, X_test = X_seqs[:split_point], X_seqs[split_point:]
    y_train, y_test = y_targets[:split_point], y_targets[split_point:]

    print(f"Train sequences: {X_train.shape[0]}, Test sequences: {X_test.shape[0]}")

    print("Normalizing features...")
    X_train, X_test = normalize_features(X_train, X_test)

    print("Building ConvLSTM model...")
    model = build_convlstm_model(SEQ_LEN, n_rows, n_cols, len(FEATURE_VARS))
    model.summary()

    print("Training...")
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train[..., np.newaxis],
        validation_split=0.15,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stop],
        verbose=1
    )

    print("Evaluating on test set...")
    y_pred_prob = model.predict(X_test)
    metrics_dict = evaluate(y_test, y_pred_prob)

    print("Test metrics:")
    for k, v in metrics_dict.items():
        print(f"  {k}: {v:.4f}")

    # Save metrics + model
    pd.DataFrame([metrics_dict]).to_excel(
        os.path.join(OUTPUT_DIR, 'convlstm_test_metrics.xlsx'), index=False
    )
    model.save(os.path.join(OUTPUT_DIR, 'convlstm_model.keras'))

    # Save training history for a learning curve plot later
    pd.DataFrame(history.history).to_csv(
        os.path.join(OUTPUT_DIR, 'training_history.csv'), index=False
    )

    print(f"Done. Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
