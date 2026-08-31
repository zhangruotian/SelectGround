"""Merge complete benchmark shards and report the common metrics."""
import argparse
import json
from pathlib import Path

from evaluate import metrics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--inputs", type=Path, nargs="+", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
settings = [json.loads(p.with_suffix(".settings.json").read_text()) for p in args.inputs]
common = [{k: v for k, v in item.items() if k not in {"output", "shard"}} for item in settings]
if not all(item == common[0] for item in common):
    raise ValueError("Shards must use the same model, benchmark, and inference settings.")
rows = [json.loads(line) for path in args.inputs for line in path.read_text().splitlines()]
if len({row["id"] for row in rows}) != len(rows):
    raise ValueError("Duplicate example IDs across shards.")
expected = {"screenspot_pro": 1581, "mmbench_gui_l2": 3594, "osworld_g": 510}[common[0]["benchmark"]]
if len(rows) != expected:
    raise ValueError(f"Expected {expected} examples, found {len(rows)}.")
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("x") as output:
    output.writelines(json.dumps(row) + "\n" for row in rows)
result = metrics(rows)
args.output.with_suffix(".metrics.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
