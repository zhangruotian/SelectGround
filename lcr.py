"""Latent Competitor Revisit on the trained SelectGround selector."""
import math
from collections import defaultdict

import torch
from PIL import Image

WEIGHTS = {
    "screenspot_pro": (.85, .275, .3),
    "mmbench_gui_l2": (.475, 1.625, .3),
    "osworld_g": (.125, .15, .3),
}
CENTERS = ((.3, .3), (.7, .3), (.3, .7), (.7, .7))


def crop_box(point, size, fraction):
    width, height = size
    w, h = min(width, max(320, round(fraction * width))), min(height, max(320, round(fraction * height)))
    left = round(min(max(0., point[0] - w / 2), width - w))
    top = round(min(max(0., point[1] - h / 2), height - h))
    return left, top, left + w, top + h


def region_score(scores, grid, size, box):
    x = (torch.arange(grid[1]) + .5) / grid[1] * size[0]
    y = (torch.arange(grid[0]) + .5) / grid[0] * size[1]
    mask = ((x[None, :] >= box[0]) & (x[None, :] < box[2])
            & (y[:, None] >= box[1]) & (y[:, None] < box[3])).flatten()
    values = scores[mask]
    return float(torch.logsumexp(values.float(), 0) - math.log(values.numel()))


def local_score(scores, grid, size, point):
    row = min(grid[0] - 1, max(0, int(point[1] / size[1] * grid[0])))
    column = min(grid[1] - 1, max(0, int(point[0] / size[0] * grid[1])))
    values = scores.reshape(grid)[max(0, row-1):min(grid[0], row+2),
                                  max(0, column-1):min(grid[1], column+2)].flatten()
    return float(torch.logsumexp(values.float(), 0) - math.log(values.numel()))


def response(point, box):
    left, top, right, bottom = box
    x, y = point
    if not left <= x < right or not top <= y < bottom:
        return None
    return f"[{round(1000*(x-left)/(right-left))},{round(1000*(y-top)/(bottom-top))}]"


@torch.inference_mode()
def score(grounder, prefix, responses):
    unique = list(dict.fromkeys(responses))
    if not unique:
        return {}
    tokenizer = grounder.processor.tokenizer
    encoded = [tokenizer(text, add_special_tokens=False)["input_ids"] for text in unique]
    first = prefix.logits.float().log_softmax(-1)
    logps = [[float(first[0, ids[0]])] for ids in encoded]
    maximum = max(map(len, encoded))
    if maximum > 1:
        continuation = torch.full(
            (len(encoded), maximum-1), tokenizer.pad_token_id, dtype=torch.long, device=grounder.device,
        )
        mask = torch.zeros_like(continuation, dtype=torch.bool)
        for i, ids in enumerate(encoded):
            continuation[i, :len(ids)-1] = torch.tensor(ids[:-1], device=grounder.device)
            mask[i, :len(ids)-1] = True
        prefix.cache.batch_repeat_interleave(len(encoded))
        positions = prefix.position_ids.repeat_interleave(len(encoded), dim=-2)
        offsets = torch.arange(maximum-1, device=grounder.device).view(*([1]*(positions.ndim-1)), -1)
        output = grounder.model(
            input_ids=continuation, past_key_values=prefix.cache,
            attention_mask=torch.cat((prefix.attention_mask.repeat(len(encoded), 1), mask.long()), 1),
            position_ids=positions[..., -1:] + 1 + offsets,
            cache_position=torch.arange(prefix.length, prefix.length+maximum-1, device=grounder.device),
            use_cache=True,
        )
        for i, ids in enumerate(encoded):
            if len(ids) > 1:
                logits = output.logits[i, :len(ids)-1].float()
                labels = torch.tensor(ids[1:], device=grounder.device)
                logps[i].extend(float(v) for v in logits.log_softmax(-1).gather(1, labels[:, None])[:, 0])
    return {text: sum(values)/len(ids)*len(ids)
            for text, ids, values in zip(unique, encoded, logps, strict=True)}


def zscore(values):
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value-mean)**2 for value in values) / len(values))
    return [(value-mean) / max(std, 1e-6) for value in values]


def select(candidates, evidence, selector, size, weights):
    eligible = [c for c in candidates if c["point"] is not None]
    by_name = {c["name"]: c for c in eligible}
    accumulated = defaultdict(list)
    for view, values in evidence.items():
        unique = {}
        for name, (text, logp) in values.items():
            if name in by_name:
                unique.setdefault(text, logp)
        if not unique:
            continue
        normalized = dict(zip(unique, zscore(list(unique.values()))))
        for name, (text, _) in values.items():
            if name in by_name and by_name[name]["source_view"] != view:
                accumulated[name].append(normalized[text])
    eligible = [c for c in eligible if accumulated[c["name"]]]
    if not eligible:
        return candidates[0]
    likelihood = zscore([sum(accumulated[c["name"]])/len(accumulated[c["name"]]) for c in eligible])
    by_name = {c["name"]: c for c in eligible}
    anchor = by_name.get("q0", by_name.get("p0", eligible[0]))["point"]

    def distance(a, b):
        return math.hypot((a[0]-b[0])/size[0], (a[1]-b[1])/size[1])

    proximity = zscore([-distance(c["point"], anchor) for c in eligible])
    selection = zscore([selector[c["name"]] for c in eligible])
    agreement = zscore([
        -sum(distance(c["point"], other["point"]) for other in eligible if other is not c)/max(1, len(eligible)-1)
        for c in eligible
    ])
    scores = [likelihood[i] + weights[0]*proximity[i] + weights[1]*selection[i] + weights[2]*agreement[i]
              for i in range(len(eligible))]
    return eligible[max(range(len(eligible)), key=lambda i: (scores[i], -i))]


def revisit(grounder, source, instruction, benchmark, ablation):
    weights = WEIGHTS[benchmark]
    full = (0, 0, *source.size)
    p0, prefix, selector, grid = grounder.observe(source, instruction, capture=True)
    candidates = [{"name": "p0", "source_view": "full", **p0}]
    prefixes = {"full": prefix}
    views = {"full": (full, source)}
    boxes = {f"grid_{i}": crop_box((x*source.width, y*source.height), source.size, .6)
             for i, (x, y) in enumerate(CENTERS)}
    ranked = sorted(boxes, key=lambda name: (-region_score(selector, grid, source.size, boxes[name]), name))
    chosen = ranked[:2]
    if ablation == "no_competitors":
        chosen = []
    elif ablation == "no_selector":
        chosen = ["grid_0", "grid_3"]
        weights = (weights[0], 0., weights[2])
    elif ablation != "full":
        raise ValueError(ablation)
    selected = {}
    if p0["point"] is not None:
        selected["q0"] = crop_box(p0["point"], source.size, .4)
    selected.update({name: boxes[name] for name in chosen})
    for name, box in selected.items():
        crop = source.crop(box)
        view = crop.resize((2*crop.width, 2*crop.height), Image.Resampling.LANCZOS)
        pred, prefixes[name], _, _ = grounder.observe(view, instruction)
        point = pred["point"]
        if point is not None:
            point = [box[0]+point[0]/2, box[1]+point[1]/2]
        candidates.append({"name": name, "source_view": name, "point": point, "raw_response": pred["raw_response"]})
        views[name] = (box, view)
    evidence = {}
    for view, (box, _) in views.items():
        visible = {c["name"]: text for c in candidates if c["point"] is not None
                   and (text := response(c["point"], box)) is not None}
        values = score(grounder, prefixes.pop(view), list(visible.values()))
        evidence[view] = {name: (text, values[text]) for name, text in visible.items()}
    selection = {c["name"]: local_score(selector, grid, source.size, c["point"])
                 for c in candidates if c["point"] is not None}
    best = select(candidates, evidence, selection, source.size, weights)
    point = best["point"]
    return {
        "method": "SelectGround+LCR", "point": point,
        "normalized_point": [1000*point[0]/source.width, 1000*point[1]/source.height] if point is not None else None,
        "raw_response": best["raw_response"], "selected_candidate": best["name"],
        "view_boxes": {name: list(box) for name, (box, _) in views.items()},
    }
