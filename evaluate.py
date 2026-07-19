import argparse
import json
from pathlib import Path

from selectground import SelectGround


def load_cases(name: str, root: Path):
    if name == "screenspot_pro":
        for annotation in sorted((root / "annotations").glob("*.json")):
            for row in json.loads(annotation.read_text()):
                yield {
                    "id": row["id"],
                    "image": root / "images" / row["img_filename"],
                    "instruction": row["instruction"],
                    "target": row["bbox"],
                    "type": "xyxy",
                    "group": row.get("group"),
                }
    elif name == "ui_vision":
        for split in ("basic", "functional", "spatial"):
            path = root / "annotations" / "element_grounding" / f"element_grounding_{split}.json"
            for index, row in enumerate(json.loads(path.read_text())):
                yield {
                    "id": f"{split}-{index}",
                    "image": root / "images" / row["image_path"],
                    "instruction": row["prompt_to_evaluate"],
                    "target": row["bbox"],
                    "type": "xyxy",
                    "group": split,
                }
    else:
        benchmark = root / "benchmark" if (root / "benchmark").is_dir() else root
        for row in json.loads((benchmark / "OSWorld-G.json").read_text()):
            if row["box_type"] == "refusal":
                continue
            yield {
                "id": row["id"],
                "image": benchmark / "images" / row["image_path"],
                "instruction": row["instruction"],
                "target": row["box_coordinates"],
                "type": row["box_type"],
                "group": None,
            }


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
        return left <= x <= right and top <= y <= bottom
    vertices = list(zip(target[0::2], target[1::2]))
    previous, inside = vertices[-1], False
    for current in vertices:
        x1, y1 = current
        x2, y2 = previous
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) <= 1e-7 and min(x1, x2) - 1e-7 <= x <= max(x1, x2) + 1e-7 and min(y1, y2) - 1e-7 <= y <= max(y1, y2) + 1e-7:
            return True
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
        previous = current
    return inside


def metrics(rows, benchmark):
    if benchmark != "ui_vision":
        return {"total": len(rows), "correct": sum(row["correct"] for row in rows), "accuracy": 100 * sum(row["correct"] for row in rows) / len(rows)}
    splits = {}
    for split in ("basic", "functional", "spatial"):
        selected = [row for row in rows if row["group"] == split]
        splits[split] = 100 * sum(row["correct"] for row in selected) / len(selected)
    return {"total": len(rows), "accuracy": sum(splits.values()) / 3, "splits": splits}


parser = argparse.ArgumentParser(description="Evaluate SelectGround on a GUI grounding benchmark.")
parser.add_argument("--model", default="ruotian/SelectGround-8B")
parser.add_argument("--benchmark", choices=("screenspot_pro", "ui_vision", "osworld_g"), required=True)
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--conground", action="store_true")
parser.add_argument("--limit", type=int)
args = parser.parse_args()

cases = list(load_cases(args.benchmark, args.data))[: args.limit]
existing = []
if args.output.exists():
    existing = [json.loads(line) for line in args.output.read_text().splitlines() if line.strip()]
done = {row["id"] for row in existing}
grounder = SelectGround(args.model)
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("a") as output:
    for number, case in enumerate(cases, 1):
        if case["id"] in done:
            continue
        prediction = grounder.predict(case["image"], case["instruction"], conground=args.conground)
        row = {
            "id": case["id"],
            "instruction": case["instruction"],
            "image": str(case["image"]),
            "point": prediction["point"],
            "correct": contains(prediction["point"], case["target"], case["type"]),
            "group": case["group"],
            "prediction": prediction,
        }
        output.write(json.dumps(row) + "\n")
        output.flush()
        existing.append(row)
        print(f"[{number}/{len(cases)}] {case['id']} correct={int(row['correct'])}", flush=True)
print(json.dumps(metrics(existing, args.benchmark), indent=2))
