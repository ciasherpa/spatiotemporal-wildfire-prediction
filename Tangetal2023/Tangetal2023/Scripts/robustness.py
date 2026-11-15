"""
robustness.py  --  is the fire-history gain stable, or did we get lucky?

Repeats the key comparison from experiments.py many times:
  - several random SEEDS (tree models and boosting change a little with the seed)
  - several TIME SPLITS (train on the first 60% / 70% / 80% of months, test on the next block)

For each run it records average precision (AP). 
The baseline is the Random Forest on last-month climate (feature set A), the closest thing to the parent paper's approach; each other model is
compared with that baseline using the SAME seed and the SAME split.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score

import experiments as E
from common import load_data, SEQ_LEN, OUTPUT_DIR

# ---------------- settings you can play with ----------------
SEEDS = [1, 2, 3]
TRAIN_FRACS = [0.6, 0.7, 0.8]      # 0.8 is the same split experiments.py uses
FEATURE_SETS = ['A last-month climate',
                'C +season +fire history',
                'D +season +fire history +neighbours']
REF_MODEL, REF_SET = 'RF', 'A last-month climate'
# ------------------------------------------------------------


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df, n_time, *_ = load_data(sys.argv[1]) if len(sys.argv) > 1 else load_data()

    feat = E.build_features(df)
    feat = feat[feat['t'] >= SEQ_LEN].reset_index(drop=True)
    sets = {k: v for k, v in E.feature_sets().items() if k in FEATURE_SETS}
    test_len = n_time - int(n_time * 0.8)        # every split tests on a block this long

    runs = []
    for frac in TRAIN_FRACS:
        cut = int(n_time * frac)
        train = feat[feat['t'] < cut]
        test = feat[(feat['t'] >= cut) & (feat['t'] < cut + test_len)]
        y = test['fire'].values
        label = f"test months {cut}-{cut + test_len - 1}"
        if y.sum() == 0 or len(train) == 0:
            print(f"Skipping {label}: no fires in the test block or no training rows")
            continue
        print(f"\n=== {label}: train rows {len(train)} (fire rate {train['fire'].mean():.1%}), "
              f"test rows {len(test)} (fire rate {y.mean():.1%}) ===")

        p = E.baseline_probs('persistence', train, test)
        runs.append(dict(split=label, seed=0, model='Persistence', features='-',
                         ap=average_precision_score(y, p)))

        for fs_name, cols in sets.items():
            med = train[cols].median()
            Xtr = train[cols].fillna(med).fillna(0)
            Xte = test[cols].fillna(med).fillna(0)
            for seed in SEEDS:
                E.SEED = seed                      # make_models reads this when it builds the models
                for mname, proto in E.make_models(train['fire'].mean()).items():
                    p = E.fit_predict(mname, clone(proto), Xtr, train['fire'], Xte)
                    runs.append(dict(split=label, seed=seed, model=mname, features=fs_name,
                                     ap=average_precision_score(y, p)))
            print(f"  done: {fs_name}")

    res = pd.DataFrame(runs)
    res.to_csv(os.path.join(OUTPUT_DIR, 'robustness_runs.csv'), index=False)

    # ---- table 1: AP as mean +- spread over seeds, one column per time split
    g = res.groupby(['model', 'features', 'split'])['ap'].agg(['mean', 'std']).reset_index()
    g['txt'] = [f"{m:.3f}+-{(0 if np.isnan(s) else s):.3f}" for m, s in zip(g['mean'], g['std'])]
    table = g.pivot(index=['model', 'features'], columns='split', values='txt')
    print("\n=== Average precision, mean +- spread over seeds, for each time split ===")
    print("(Persistence = 'fire next month = fire last month'; it has no seeds)")
    print(table.to_string())

    # ---- table 2: gain over the baseline, same seed and same split
    ref = res[(res['model'] == REF_MODEL) & (res['features'] == REF_SET)] \
        .set_index(['split', 'seed'])['ap']
    cmp_ = res[res['model'] != 'Persistence'].copy()
    cmp_['ref_ap'] = [ref.get((s, sd), np.nan) for s, sd in zip(cmp_['split'], cmp_['seed'])]
    cmp_['gain'] = cmp_['ap'] - cmp_['ref_ap']
    cmp_ = cmp_[~((cmp_['model'] == REF_MODEL) & (cmp_['features'] == REF_SET))]
    summ = (cmp_.groupby(['model', 'features'])
            .agg(mean_gain=('gain', 'mean'), worst_gain=('gain', 'min'),
                 share_of_runs_better=('gain', lambda x: (x > 0).mean()),
                 n_runs=('gain', 'size'))
            .reset_index().sort_values('mean_gain', ascending=False))
    summ.to_csv(os.path.join(OUTPUT_DIR, 'robustness_gain_summary.csv'), index=False)
    print(f"\n=== Gain in average precision over '{REF_MODEL} | {REF_SET}' "
          f"(same seed, same split; every run counts) ===")
    print("worst_gain > 0 means the model beat the baseline in EVERY run.")
    print(summ.round(3).to_string(index=False))


if __name__ == '__main__':
    main()