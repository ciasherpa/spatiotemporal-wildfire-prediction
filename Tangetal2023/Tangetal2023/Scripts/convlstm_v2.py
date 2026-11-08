"""
convlstm_v2.py  --  ConvLSTM, version 2

What's different from the first run:
  1. Evaluated ONLY on real (location, month) cells, not the empty padded grid cells.
  2. The loss ignores empty cells too (the model no longer "learns" from padding).
  3. Weighted loss: missing a fire costs `pos_weight` times more than a false alarm.
     We train once per weight in POS_WEIGHTS so you can show the tradeoff.
  4. Same train/test months as rf_baseline.py (see common.py), so results are comparable.
  5. Normalization uses only real cells from the training months.
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from common import (load_data, build_grid, evaluate_thresholds,
                    FEATURE_VARS, SEQ_LEN, OUTPUT_DIR)

# ---------------- settings you can play with ----------------
POS_WEIGHTS = [1, 5, 10, 20]   # how much more a missed fire costs (1 = no weighting)
EPOCHS = 40
BATCH_SIZE = 8
FILTERS = (16, 8)              # size of the two ConvLSTM layers
SEED = 1
# ------------------------------------------------------------


def make_sequences(X_norm, y, valid, seq_len):
    """Window of seq_len months -> fire labels of the following month."""
    Xs, Ys, target_t = [], [], []
    for t in range(seq_len, X_norm.shape[0]):
        Xs.append(X_norm[t - seq_len:t])
        # channel 0 = fire label, channel 1 = "is this a real cell" (used by loss + evaluation)
        Ys.append(np.stack([y[t], valid[t].astype(np.float32)], axis=-1))
        target_t.append(t)
    return np.stack(Xs), np.stack(Ys), np.array(target_t)


def masked_weighted_bce(pos_weight):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        labels, real = y_true[..., 0:1], y_true[..., 1:2]
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        bce = -(pos_weight * labels * tf.math.log(y_pred)
                + (1 - labels) * tf.math.log(1 - y_pred))
        return tf.reduce_sum(bce * real) / (tf.reduce_sum(real) + 1e-7)
    return loss


def build_model(n_rows, n_cols, n_features):
    f1, f2 = FILTERS
    return models.Sequential([
        layers.Input(shape=(SEQ_LEN, n_rows, n_cols, n_features)),
        layers.ConvLSTM2D(f1, (3, 3), padding='same', return_sequences=True, activation='tanh'),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(f2, (3, 3), padding='same', return_sequences=False, activation='tanh'),
        layers.BatchNormalization(),
        layers.Conv2D(1, (1, 1), activation='sigmoid'),
    ])


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df, n_time, n_rows, n_cols, cutoff_t = load_data()
    X_grid, y_grid, valid = build_grid(df, n_time, n_rows, n_cols)

    # normalize with TRAIN months + real cells only; empty cells become 0
    train_part = X_grid[:cutoff_t]
    mean = np.nanmean(train_part, axis=(0, 1, 2))
    std = np.nanstd(train_part, axis=(0, 1, 2))
    std[std < 1e-8] = 1.0
    X_norm = np.nan_to_num((X_grid - mean) / std, nan=0.0).astype(np.float32)

    X_seq, Y_pack, target_t = make_sequences(X_norm, y_grid, valid, SEQ_LEN)
    is_train = target_t < cutoff_t
    X_train, Y_train = X_seq[is_train], Y_pack[is_train]
    X_test, Y_test = X_seq[~is_train], Y_pack[~is_train]
    print(f"Train sequences: {len(X_train)} | Test sequences: {len(X_test)}")
    if len(X_train) < 20 or len(X_test) < 5:
        raise ValueError("Too few sequences -- check the month counts printed above.")

    real_test = Y_test[..., 1] > 0.5
    y_true = Y_test[..., 0][real_test]
    print(f"Test cells scored: {real_test.sum()} (fire rate {y_true.mean():.2%})")

    results = []
    for pw in POS_WEIGHTS:
        tf.keras.backend.clear_session()
        np.random.seed(SEED)
        tf.random.set_seed(SEED)

        model = build_model(n_rows, n_cols, len(FEATURE_VARS))
        model.compile(optimizer='adam', loss=masked_weighted_bce(pw))
        stop = callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
        hist = model.fit(X_train, Y_train, validation_split=0.15, batch_size=BATCH_SIZE,
                         epochs=EPOCHS, callbacks=[stop], verbose=0)

        probs = model.predict(X_test, verbose=0)[..., 0][real_test]
        for row in evaluate_thresholds(y_true, probs):
            row['model'] = f'ConvLSTM (pos_weight={pw})'
            row['n_features'] = len(FEATURE_VARS)
            results.append(row)
        best = max((r for r in results if r['model'] == f'ConvLSTM (pos_weight={pw})'),
                   key=lambda r: r['f1'])
        print(f"pos_weight={pw:3d}  epochs={len(hist.history['loss']):2d}  "
              f"avg_precision={best['avg_precision']:.3f}  best F1={best['f1']:.3f} "
              f"(threshold {best['threshold']})")

    path = os.path.join(OUTPUT_DIR, 'convlstm_results.csv')
    pd.DataFrame(results).to_csv(path, index=False)
    print(f"Saved: {path}")


if __name__ == '__main__':
    main()
