"""Success rate per task (<output dir>/<task>/simulation_results.csv written by rollout_policy.py) and the mean."""
import csv
import pathlib
import sys

rates = {}
for f in sorted(pathlib.Path(sys.argv[1]).glob("*/simulation_results.csv")):
    rows = list(csv.DictReader(open(f)))
    rates[f.parent.name] = sum(int(r["success"]) for r in rows) / max(len(rows), 1)
for name, rate in rates.items():
    print(f"{rate:6.1%}  {name}")
print(f"{sum(rates.values()) / max(len(rates), 1):6.1%}  mean over {len(rates)} tasks")
