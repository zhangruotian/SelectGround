from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, get_scheduler

from selectground import PROMPT, _attention_logits, _find_config, _value


RECIPES = {
    "8b": {
        "base": "Qwen/Qwen3-VL-8B-Instruct",
        "revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "data": "ruotian/SelectGround-Data",
        "steps": 135,
        "gpus": 2,
        "accumulation": 64,
        "learning_rate": 5e-5,
    },
    "30b": {
        "base": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "revision": "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c",
        "data": "ruotian/SelectGround-Data",
        "steps": 200,
        "gpus": 4,
        "accumulation": 4,
        "learning_rate": 4e-5,
    },
}
LAYERS = list(range(18, 24))
SEED = 20260625


class Rows(Dataset):
    def __init__(self, rows: list[dict[str, Any]], root: Path) -> None:
        self.rows, self.root = rows, root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {**self.rows[index], "image": str(self.root / self.rows[index]["image"])}


class HeadSelector(nn.Module):
    def __init__(self, heads: int) -> None:
        super().__init__()
        self.layer_head_weights = nn.Parameter(torch.zeros(len(LAYERS), heads))

    def forward(self, values: list[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack([value.float() for value in values])
        weights = self.layer_head_weights.flatten().softmax(0).view_as(self.layer_head_weights).to(stacked.device)
        return (stacked * weights[:, :, None]).sum(dim=(0, 1))


class Attention:
    def __init__(self, model: Any, query: int, visual: torch.Tensor) -> None:
        self.model, self.query, self.visual = model, query, visual
        self.values: dict[int, torch.Tensor] = {}
        self.handles: list[Any] = []

    def __enter__(self) -> "Attention":
        for module in self.model.modules():
            layer = getattr(module, "layer_idx", None)
            if layer in LAYERS and hasattr(module, "q_proj"):
                self.handles.append(module.register_forward_hook(self._hook(int(layer)), with_kwargs=True))
        return self

    def __exit__(self, *_: Any) -> None:
        for handle in self.handles:
            handle.remove()

    def _hook(self, layer: int):
        def hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden is not None and hidden.shape[1] > self.query:
                self.values[layer] = _attention_logits(
                    module, hidden, self.query, self.visual, kwargs["position_embeddings"]
                )

        return hook

    def ordered(self) -> list[torch.Tensor]:
        return [self.values[layer] for layer in LAYERS]


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def loader(rows: list[dict[str, Any]], root: Path, seed: int) -> DataLoader:
    return DataLoader(
        Rows(rows, root),
        batch_size=1,
        shuffle=True,
        collate_fn=lambda batch: batch[0],
        generator=torch.Generator().manual_seed(seed),
    )


def next_row(data_loader: DataLoader, iterator: Any):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(data_loader)
        return next(iterator), iterator


def encode(processor: Any, row: dict[str, Any], device: torch.device):
    from qwen_vl_utils import process_vision_info

    user = {
        "role": "user",
        "content": [
            {"type": "image", "image": row["image"]},
            {"type": "text", "text": PROMPT.format(instruction=row["instruction"])},
        ],
    }
    prompt = [user]
    full = [user, {"role": "assistant", "content": [{"type": "text", "text": row["response"]}]}]

    def process(messages: list[dict[str, Any]], generation_prompt: bool):
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation_prompt)
        images, videos = process_vision_info(messages)
        kwargs = {"text": [text], "images": images, "padding": True, "return_tensors": "pt"}
        if videos is not None:
            kwargs["videos"] = videos
        return processor(**kwargs).to(device)

    inputs = process(full, False)
    prompt_length = int(process(prompt, True)["attention_mask"].sum())
    labels = inputs["input_ids"].clone()
    labels[:, :prompt_length] = -100
    return inputs, labels, prompt_length - 1


def coordinate_loss(logits: torch.Tensor, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    start = input_ids.shape[1] - logits.shape[1]
    targets = input_ids[:, start + 1 :]
    mask = labels[:, start + 1 :].ne(-100)
    token_logps = logits[:, :-1].float().log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return -(token_logps * mask).sum().to(logits.dtype) / mask.sum()


def box_mask(box: list[float], row: dict[str, Any], grid: torch.Tensor, config: Any) -> torch.Tensor:
    vision = _value(config, "vision_config")
    patch, merge = int(_value(vision, "patch_size", 16)), int(_value(vision, "spatial_merge_size", 2))
    grid = grid.detach().cpu().long()
    height, width = int(grid[1]) // merge, int(grid[2]) // merge
    resized_width, resized_height = int(grid[2]) * patch, int(grid[1]) * patch
    x1, y1, x2, y2 = box
    left, right = sorted((x1 / row["image_width"] * resized_width, x2 / row["image_width"] * resized_width))
    top, bottom = sorted((y1 / row["image_height"] * resized_height, y2 / row["image_height"] * resized_height))
    rows = torch.arange(height)[:, None]
    columns = torch.arange(width)[None, :]
    return (
        (left < (columns + 1) * resized_width / width)
        & (right > columns * resized_width / width)
        & (top < (rows + 1) * resized_height / height)
        & (bottom > rows * resized_height / height)
    ).flatten()


def region_score(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = scores[mask.to(scores.device)]
    return torch.logsumexp(selected.float(), 0) - math.log(selected.numel())


def selection_loss(scores: torch.Tensor, row: dict[str, Any], grid: torch.Tensor, config: Any) -> torch.Tensor:
    target = box_mask(row["target_bbox"], row, grid, config)
    distractor = box_mask(row["distractor_bbox"], row, grid, config)
    overlap = target & distractor
    target, distractor = target & ~overlap, distractor & ~overlap
    if not target.any() or not distractor.any():
        return scores.sum() * 0
    target_score, distractor_score = region_score(scores, target), region_score(scores, distractor)
    candidates = [target, distractor]
    extras = []
    for index, box in enumerate(row.get("candidate_bboxes", [])):
        mask = box_mask(box, row, grid, config)
        if mask.any() and not (mask & target).any() and not (mask & distractor).any():
            extras.append((float(region_score(scores, mask).detach()), -index, mask))
    extras.sort(reverse=True, key=lambda item: item[:2])
    candidates.extend(item[2] for item in extras[:3])
    listwise = torch.logsumexp(torch.stack([region_score(scores, mask) for mask in candidates]), 0) - target_score
    pair = F.softplus(scores.new_tensor(.3) - target_score + distractor_score)
    return listwise + .5 * pair


def warmup_cosine(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / warmup
    if step >= total:
        return 0.0
    return .5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))


def scheduler_for(model_size: str, optimizer: torch.optim.Optimizer):
    if model_size == "30b":
        return get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=10, num_training_steps=200)
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    target_lrs = [1e-6, 1e-4]
    functions = []
    for base, target in zip(base_lrs, target_lrs):
        def schedule(step: int, base=base, target=target):
            if step < 126:
                return warmup_cosine(step, 10, 200)
            return target / base * warmup_cosine(step - 125, 10, 25)
        functions.append(schedule)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, functions)


def save(
    accelerator: Accelerator,
    model: Any,
    selector: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    processor: Any,
    output: Path,
    revision: str,
    completed: int,
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(output, safe_serialization=True)
        config_path = output / "adapter_config.json"
        config = json.loads(config_path.read_text())
        config["revision"] = revision
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        merger = {name: value.detach().cpu() for name, value in unwrapped.named_parameters() if ".visual.merger." in f".{name}"}
        torch.save({"state_dict": merger}, output / "visual_merger.pt")
        head = accelerator.unwrap_model(selector)
        torch.save({"layers": LAYERS, "layer_head_weights": head.layer_head_weights.detach().cpu()}, output / "selection_head.pt")
        torch.save(
            {"completed": completed, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
            output / "training_state.pt",
        )
        processor.save_pretrained(output)
    accelerator.wait_for_everyone()
    rng = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }
    torch.save(rng, output / f"rng_state_rank_{accelerator.process_index}.pt")


def restore(checkpoint: Path, optimizer: torch.optim.Optimizer, scheduler: Any, accelerator: Accelerator) -> int:
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    rng = torch.load(
        checkpoint / f"rng_state_rank_{accelerator.process_index}.pt",
        map_location="cpu",
        weights_only=False,
    )
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    torch.cuda.set_rng_state_all(rng["cuda"])
    return int(state["completed"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the two SelectGround checkpoints from the paper.")
    parser.add_argument("--model", choices=("8b", "30b"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "phase_a", "phase_b"), default="all")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    recipe = RECIPES[args.model]
    if args.model == "8b" and args.stage == "all":
        raise ValueError("8b training requires --stage phase_a followed by --stage phase_b")
    if args.model == "30b" and args.stage != "all":
        raise ValueError("30b training uses --stage all")
    if args.stage == "phase_b" and args.checkpoint is None:
        raise ValueError("phase_b requires --checkpoint")
    accelerator = Accelerator(
        gradient_accumulation_steps=recipe["accumulation"],
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )
    if accelerator.num_processes != recipe["gpus"]:
        raise ValueError(f"{args.model} training requires {recipe['gpus']} processes")
    set_seed(SEED + accelerator.process_index)
    with accelerator.main_process_first():
        data = Path(snapshot_download(recipe["data"], repo_type="dataset"))
    pairs = read_rows(data / "data" / "train_pairs.jsonl")
    random.Random(SEED).shuffle(pairs)
    pairs = pairs[max(1, round(.02 * len(pairs))) :]
    replay = read_rows(data / "data" / "train_replay.jsonl")
    pair_seed, replay_seed = SEED + 1001, SEED + 2001
    if args.stage == "phase_b":
        pairs = read_rows(data / "data" / "refinement_pairs.jsonl")
        replay = read_rows(data / "data" / "refinement_replay.jsonl")
        pair_seed, replay_seed = SEED + 3001, SEED + 4001
    pair_loader, replay_loader = loader(pairs, data, pair_seed), loader(replay, data, replay_seed)

    processor = AutoProcessor.from_pretrained(
        recipe["base"], revision=recipe["revision"], min_pixels=3136, max_pixels=8847360
    )
    base_config = AutoConfig.from_pretrained(recipe["base"], revision=recipe["revision"])
    model = AutoModelForImageTextToText.from_pretrained(
        recipe["base"], revision=recipe["revision"], config=base_config,
        dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    if args.checkpoint is None:
        model = get_peft_model(model, LoraConfig(
            r=64,
            lora_alpha=128,
            lora_dropout=.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        ))
    else:
        import transformers.integrations.tensor_parallel as tensor_parallel

        if not hasattr(tensor_parallel, "EmbeddingParallel"):
            tensor_parallel.EmbeddingParallel = type("EmbeddingParallel", (), {})
        model = PeftModel.from_pretrained(model, args.checkpoint, is_trainable=True)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.bfloat16)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    config = _find_config(model)
    heads = int(_value(_value(config, "text_config", config), "num_attention_heads"))
    selector = HeadSelector(heads)
    if args.checkpoint is not None:
        head = torch.load(args.checkpoint / "selection_head.pt", map_location="cpu", weights_only=False)
        selector.layer_head_weights.data.copy_(head["layer_head_weights"])
    optimizer = torch.optim.AdamW([
        {"params": [parameter for parameter in model.parameters() if parameter.requires_grad], "lr": recipe["learning_rate"]},
        {"params": selector.parameters(), "lr": 1e-4},
    ], weight_decay=0.0)
    scheduler = scheduler_for(args.model, optimizer)
    model, selector, optimizer, pair_loader, replay_loader = accelerator.prepare(
        model, selector, optimizer, pair_loader, replay_loader
    )
    model.train()
    selector.train()
    iterators = [iter(pair_loader), iter(replay_loader)]
    micro_step = 0
    completed = restore(args.checkpoint, optimizer, scheduler, accelerator) if args.checkpoint else 0
    target = 125 if args.stage == "phase_a" else recipe["steps"]
    optimizer.zero_grad(set_to_none=True)
    while completed < target:
        active_loaders, active_iterators = (pair_loader, replay_loader), iterators
        semantic = micro_step % 2 == 1
        index = 0 if semantic else 1
        row, active_iterators[index] = next_row(active_loaders[index], active_iterators[index])
        with accelerator.accumulate(model, selector):
            inputs, labels, query = encode(processor, row, accelerator.device)
            visual = torch.nonzero(inputs["input_ids"][0] == int(_value(config, "image_token_id")), as_tuple=False).flatten()
            keep = int(labels.ne(-100).sum()) + 1
            context = Attention(model, query, visual) if semantic else nullcontext()
            with context as attention:
                output = model(**inputs, use_cache=False, logits_to_keep=keep)
                if semantic:
                    scores = selector(attention.ordered())
                    semantic_loss = selection_loss(scores, row, inputs["image_grid_thw"][0], config)
                else:
                    semantic_loss = output.logits.sum() * 0
                coord_loss = coordinate_loss(output.logits, inputs["input_ids"], labels)
                loss = coord_loss + .1 * semantic_loss
                accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(list(model.parameters()) + list(selector.parameters()), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        micro_step += 1
        if accelerator.sync_gradients:
            completed += 1
            if accelerator.is_main_process:
                print(f"step={completed} loss={float(loss):.4f} coord={float(coord_loss):.4f} selection={float(semantic_loss):.4f}", flush=True)
    save(
        accelerator,
        model,
        selector,
        optimizer,
        scheduler,
        processor,
        args.output,
        recipe["revision"],
        completed,
    )


if __name__ == "__main__":
    main()
