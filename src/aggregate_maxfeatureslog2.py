"""Aggregates the 3-seed ExtraTrees max_features=log2 rerun (see
run_phase2_maxfeatureslog2.ps1) into the same summary format as
run_all_seeds.py's baseline aggregation."""
import json
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parent.parent / "runs"
SEEDS = [1, 2, 3]

all_results = []
for seed in SEEDS:
    with open(RUNS / f"seed{seed}" / "maxfeatureslog2" / "phase2_results.json") as f:
        all_results.append(json.load(f))

names = list(all_results[0].keys())
summary = {}
for name in names:
    accs = [r[name]["accuracy"] for r in all_results]
    precs = [r[name]["precision_macro"] for r in all_results]
    recs = [r[name]["recall_macro"] for r in all_results]
    f1s = [r[name]["f1_macro"] for r in all_results]
    summary[name] = {
        "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
        "precision_mean": float(np.mean(precs)), "precision_std": float(np.std(precs)),
        "recall_mean": float(np.mean(recs)), "recall_std": float(np.std(recs)),
        "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
    }

with open(RUNS / "summary_3seeds_maxfeatureslog2.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== 3-seed max_features=log2 summary (mean +/- std, %) ===")
for name, s in summary.items():
    print(f"{name:20s} acc={s['accuracy_mean']*100:.2f}+/-{s['accuracy_std']*100:.2f}  "
          f"prec={s['precision_mean']*100:.2f}+/-{s['precision_std']*100:.2f}  "
          f"rec={s['recall_mean']*100:.2f}+/-{s['recall_std']*100:.2f}  "
          f"f1={s['f1_mean']*100:.2f}+/-{s['f1_std']*100:.2f}")
print(f"\nsaved: {RUNS / 'summary_3seeds_maxfeatureslog2.json'}")
