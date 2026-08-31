"""Evaluate the three paper benchmarks with strict annotation geometry."""
import argparse
import io
import itertools
import json
import math
from collections import Counter
from pathlib import Path

import pyarrow.parquet as parquet
from PIL import Image


BENCHMARKS = ("screenspot_pro", "mmbench_gui_l2", "osworld_g")


def load_cases(name, root):
    if name == "screenspot_pro":
        paths = sorted((root / "data").glob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No ScreenSpot-Pro parquet files in {root / 'data'}")
        for path in paths:
            for batch in parquet.ParquetFile(path).iter_batches(batch_size=1):
                row = batch.to_pylist()[0]
                yield {
                    "id": str(row["id"]), "image": Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB"),
                    "instruction": row["instruction"], "target": row["bbox"], "type": "xyxy",
                }
    elif name == "mmbench_gui_l2":
        for row in json.loads((root / "L2_annotations.json").read_text()):
            width, height = row["image_size"]
            x1, y1, x2, y2 = row["bbox"]
            yield {
                "id": str(row["index"]), "image": root / "offline_images" / row["platform"] / row["image_path"],
                "instruction": row["instruction"], "target": [x1*width, y1*height, x2*width, y2*height], "type": "xyxy",
            }
    elif name == "osworld_g":
        for row in json.loads((root / "benchmark/OSWorld-G.json").read_text()):
            if row["box_type"] != "refusal":
                yield {
                    "id": str(row["id"]), "image": root / "benchmark/images" / row["image_path"],
                    "instruction": row["instruction"], "target": row["box_coordinates"], "type": row["box_type"],
                }
    else:
        raise ValueError(name)


def contains(point, target, target_type):
    if point is None:
        return False
    x, y = point
    if target_type in {"bbox", "xyxy"}:
        if target_type == "xyxy":
            left, top, right, bottom = target
        else:
            left, top, width, height = target[:4]
            right, bottom = left + width, top + height
        center_x, center_y = (left + right) / 2, (top + bottom) / 2
        half_width, half_height = abs(right - left) / 2, abs(bottom - top) / 2
        return (
            center_x - half_width <= x <= center_x + half_width
            and center_y - half_height <= y <= center_y + half_height
        )
    vertices = list(zip(target[0::2], target[1::2]))
    previous, inside = vertices[-1], False
    for current in vertices:
        x1, y1 = current
        x2, y2 = previous
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
        previous = current
    return inside



def classify(point, case, size):
    target = case["target"]
    if contains(point, target, case["type"]):
        return "correct"
    if point is None or not all(math.isfinite(v) for v in point):
        return "others"
    if not (0 <= point[0] <= size[0] and 0 <= point[1] <= size[1]):
        return "others"
    if case["type"] == "polygon":
        xs, ys = target[0::2], target[1::2]
        left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
    elif case["type"] == "bbox":
        left, top, width, height = target[:4]
        right, bottom = left + width, top + height
    else:
        left, top, right, bottom = target
    cx, cy = (left+right)/2, (top+bottom)/2
    return "localization_miss" if (
        cx-1.5*(right-left) <= point[0] <= cx+1.5*(right-left)
        and cy-1.5*(bottom-top) <= point[1] <= cy+1.5*(bottom-top)
    ) else "semantic_miss"


def metrics(rows):
    counts = Counter(row["label"] for row in rows)
    total = len(rows)
    return {
        "total": total, **{key: counts[key] for key in ["correct", "semantic_miss", "localization_miss", "others"]},
        "accuracy": 100*counts["correct"]/total, "semantic_miss_rate": 100*counts["semantic_miss"]/total,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ruotian/SelectGround-8B")
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lcr", action="store_true")
    parser.add_argument("--ablation", choices=("full", "no_competitors", "no_selector"), default="full")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.shard < args.num_shards:
        parser.error("--shard must be in [0, --num-shards)")
    if args.ablation != "full" and not args.lcr:
        parser.error("--ablation requires --lcr")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    settings_path = args.output.with_suffix(".settings.json")
    settings = json.loads(json.dumps(vars(args), default=str))
    if settings_path.exists():
        if json.loads(settings_path.read_text()) != settings:
            raise ValueError("The existing output belongs to a different evaluation configuration.")
    else:
        if args.output.exists():
            raise FileExistsError("Output exists without evaluation settings; choose a new output.")
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    rows = [json.loads(line) for line in args.output.read_text().splitlines()] if args.output.exists() else []
    done = {row["id"] for row in rows}
    cases = (case for i, case in enumerate(load_cases(args.benchmark, args.data)) if i % args.num_shards == args.shard)
    if args.limit is not None:
        cases = itertools.islice(cases, args.limit)
    from selectground import SelectGround
    grounder = SelectGround(args.model)
    with args.output.open("a") as output:
        for case in cases:
            if case["id"] in done:
                continue
            image = case["image"] if isinstance(case["image"], Image.Image) else Image.open(case["image"]).convert("RGB")
            pred = grounder.predict(image, case["instruction"], lcr=args.lcr,
                                    benchmark=args.benchmark, ablation=args.ablation)
            row = {"id": case["id"], **pred, "label": classify(pred["point"], case, image.size)}
            rows.append(row)
            output.write(json.dumps(row) + "\n")
            output.flush()
    result = metrics(rows)
    args.output.with_suffix(".metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
