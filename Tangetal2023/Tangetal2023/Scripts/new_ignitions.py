"""
overall question: does the fire-history gain hold for NEW fires?

experiments.py scores every test row. But a lot of "fire last month -> fire this month" may just be
a fire that kept burning. 

This code scores ONLY the test rows where there was NO fire last month, so a
continuing fire cannot help the model. Same data, same models, same train/test split as experiments.py.

Run (from the Scripts folder): python new_ignitions.py
Output: experiments_new_ignitions.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score

import experiments as E
from common import load_data, SEQ_LEN, OUTPUT_DIR

REF_NAME = 'RF | A last-month climate'   # the parent-paper-style baseline


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df, n_time, n_rows, n_cols, cutoff_t = load_data(sys.argv[1]) if len(sys.argv) > 1 else load_data()

    feat = E.build_features(df)
    feat = feat[feat['t'] >= SEQ_LEN].reset_index(drop=True)
    train = feat[feat['t'] < cutoff_t]
    test = feat[feat['t'] >= cutoff_t]

    mask = (test['fire_lag1'] == 0).values        # there was a record last month and it had no fire
    y = test['fire'].values[mask]
    months = test['t'].values[mask]
    print(f"\nTest rows with NO fire last month: {mask.sum()} of {len(mask)}")
    print(f"Fire rate among them: {y.mean():.3f}   (a no-skill model scores about this)")
    if y.sum() == 0:
        raise SystemExit("No fires among the 'no fire last month' rows, so there is nothing to score.")

    probs = {}
    for kind, label in [('clim_month', 'Climatology (calendar month)'),
                        ('clim_loc', 'Climatology (location)')]:
        probs[label] = E.baseline_probs(kind, train, test)[mask]

    for fs_name, cols in E.feature_sets().items():
        med = train[cols].median()
        Xtr = train[cols].fillna(med).fillna(0)
        Xte = test[cols].fillna(med).fillna(0)
        for mname, proto in E.make_models(train['fire'].mean()).items():
            p = E.fit_predict(mname, clone(proto), Xtr, train['fire'], Xte)
            probs[f'{mname} | {fs_name}'] = p[mask]
            print(f"  fitted {mname} | {fs_name}")

    rows = []
    for name, p in probs.items():
        lo, hi = E.month_bootstrap_ap(y, p, months)
        row = {'model': name, 'ap_new_ignitions': average_precision_score(y, p),
               'ap_lo': lo, 'ap_hi': hi}
        if REF_NAME in probs and name != REF_NAME:
            d, glo, ghi, share = E.paired_month_bootstrap(y, probs[REF_NAME], p, months)
            row.update(gain_vs_baseline=d, gain_lo=glo, gain_hi=ghi)
        rows.append(row)

    res = pd.DataFrame(rows).sort_values('ap_new_ignitions', ascending=False)
    path = os.path.join(OUTPUT_DIR, 'experiments_new_ignitions.csv')
    res.to_csv(path, index=False)
    print(f"\nSaved: {path}")

    ref_ap = res.loc[res['model'] == REF_NAME, 'ap_new_ignitions']
    if len(ref_ap):
        print(f"\nBaseline '{REF_NAME}' scores AP={ref_ap.iloc[0]:.3f} on these rows.")
    print("=== New fires only: models ranked by average precision "
          "(gain columns = improvement over the baseline; interval from resampling test months) ===")
    print(res.round(3).head(15).to_string(index=False))

if __name__ == '__main__':
    main()