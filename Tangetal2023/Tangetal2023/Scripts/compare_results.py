"""
compare_results.py  --  merges rf_results.csv + convlstm_results.csv into one table.
Run AFTER rf_baseline.py and convlstm_v2.py.
"""
import os
import pandas as pd
from common import OUTPUT_DIR

parts = []
for f in ['rf_results.csv', 'convlstm_results.csv']:
    p = os.path.join(OUTPUT_DIR, f)
    if os.path.exists(p):
        parts.append(pd.read_csv(p))
    else:
        print(f"(skipping {f} -- not found, run its script first)")
if not parts:
    raise SystemExit("No result files found.")

all_rows = pd.concat(parts, ignore_index=True)

# one row per model: the cutoff that gave the best F1
best = (all_rows.sort_values('f1', ascending=False)
        .groupby('model', sort=False).head(1)
        .sort_values('avg_precision', ascending=False))
summary = best[['model', 'avg_precision', 'precision', 'recall', 'f1', 'threshold',
                'n_features', 'test_fire_rate']].round(3)

out = os.path.join(OUTPUT_DIR, 'comparison_table.xlsx')
with pd.ExcelWriter(out) as w:
    summary.to_excel(w, sheet_name='summary', index=False)
    all_rows.round(4).to_excel(w, sheet_name='all_thresholds', index=False)

print(summary.to_string(index=False))
print(f"\nSaved: {out}")
print("Note: avg_precision doesn't depend on the cutoff; precision/recall/F1 are at each model's best-F1 cutoff.")
print(f"A model with no skill would score avg_precision ~ the fire rate (test_fire_rate column).")
