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
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, get_scheduler

from selectground import PROMPT


BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
BASE_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
LAYERS = list(range(18, 24))
SEED = 20260625


def _value(obj: Any, name: str, default: Any = None) -> Any:
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _find_config(model: Any) -> Any:
    config = getattr(model, "config", None)
    if config is None or _value(config, "vision_config") is None:
        raise ValueError("Could not find the Qwen3-VL model config")
    return config


def _repeat_key_value_heads(key_states: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return key_states
    batch, heads, sequence, head_dim = key_states.shape
    return (
        key_states[:, :, None, :, :]
        .expand(batch, heads, groups, sequence, head_dim)
        .reshape(batch, heads * groups, sequence, head_dim)
    )


def _attention_logits(
    attention: Any,
    hidden_states: torch.Tensor,
    query_position: int,
    visual_positions: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
) -> torch.Tensor:
    if position_embeddings is None:
        raise RuntimeError("Qwen3-VL semantic logits require position_embeddings")
    head_dim = int(attention.head_dim)
    hidden_shape = (*hidden_states.shape[:-1], -1, head_dim)
    query_states = attention.q_norm(attention.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = attention.k_norm(attention.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, *position_embeddings)
    key_states = _repeat_key_value_heads(key_states, int(getattr(attention, "num_key_value_groups", 1)))
    positions = visual_positions.to(device=hidden_states.device, dtype=torch.long)
    query = query_states[:, :, int(query_position), :]
    visual_keys = key_states.index_select(2, positions)
    logits = (query.unsqueeze(2) * visual_keys).sum(dim=-1) * float(getattr(attention, "scaling", 1.0))
    return logits.squeeze(0)


def load_visual_merger(model: Any, checkpoint: Path) -> None:
    merger_path = checkpoint / "visual_merger.pt"
    if not merger_path.is_file():
        raise FileNotFoundError(f"Missing visual merger checkpoint: {merger_path}")
    merger = torch.load(merger_path, map_location="cpu", weights_only=False)
    parameters = dict(model.named_parameters())
    state = merger.get("state_dict", merger)
    missing = sorted(set(state) - set(parameters))
    if missing:
        raise KeyError(f"Visual merger parameters missing from model: {missing[:3]}")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name].device, parameters[name].dtype))


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


def selection_loss(
    scores: torch.Tensor,
    row: dict[str, Any],
    grid: torch.Tensor,
    config: Any,
    margin: float,
    pair_weight: float,
) -> torch.Tensor:
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
    pair = F.softplus(scores.new_tensor(margin) - target_score + distractor_score)
    return listwise + pair_weight * pair


def scheduler_for(optimizer: torch.optim.Optimizer, warmup_steps: int, training_steps: int):
    return get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=training_steps,
    )

def save(
    accelerator: Accelerator,
    model: Any,
    selector: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    processor: Any,
    output: Path,
    completed: int,
    stage: str,
    micro_step: int,
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(output, safe_serialization=True)
        config_path = output / "adapter_config.json"
        config = json.loads(config_path.read_text())
        config["revision"] = BASE_REVISION
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        merger = {name: value.detach().cpu() for name, value in unwrapped.named_parameters() if ".visual.merger." in f".{name}"}
        torch.save({"state_dict": merger}, output / "visual_merger.pt")
        head = accelerator.unwrap_model(selector)
        torch.save({"layers": LAYERS, "layer_head_weights": head.layer_head_weights.detach().cpu()}, output / "selection_head.pt")
        torch.save(
            {
                "completed": completed,
                "micro_step": micro_step,
                "stage": stage,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
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
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        (output / "checkpoint_complete").write_text("complete\n", encoding="utf-8")
    accelerator.wait_for_everyone()


def restore(
    checkpoint: Path,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    accelerator: Accelerator,
    stage: str,
    accumulation: int,
) -> tuple[int, int]:
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
    completed = int(state["completed"])
    saved_stage = state.get("stage")
    micro_step = int(state.get("micro_step", completed * accumulation)) if saved_stage == stage else 0
    return completed, micro_step


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce SelectGround-8B training.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--stage", choices=("main", "refinement"), default="main")
    parser.add_argument("--steps", type=int, required=True)
    args = parser.parse_args()
    if args.checkpoint is not None and args.initialize_from is not None:
        raise ValueError("Use only one of --checkpoint and --initialize-from")
    learning_rate, auxiliary_weight, pair_every, scheduler_steps = (
        (3.45e-5, .1, 2, 320)
        if args.stage == "main"
        else (1e-6, .05, 3, 30)
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=64,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )
    if accelerator.num_processes != 2:
        raise ValueError(f"Expected 2 processes, got {accelerator.num_processes}")
    set_seed(SEED + accelerator.process_index)
    data = args.data
    pair_file, replay_file = (
        ("all_pairs.jsonl", "all_replay.jsonl")
        if args.stage == "main"
        else ("refinement_pairs.jsonl", "refinement_replay.jsonl")
    )
    pairs = read_rows(data / "data" / pair_file)
    replay = read_rows(data / "data" / replay_file)
    seed_offset = 1000 if args.stage == "main" else 3000
    pair_seed, replay_seed = SEED + seed_offset + 1, SEED + seed_offset + 1001
    pair_loader, replay_loader = loader(pairs, data, pair_seed), loader(replay, data, replay_seed)

    processor = AutoProcessor.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, min_pixels=3136, max_pixels=8847360
    )
    base_config = AutoConfig.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, config=base_config,
        dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    source_checkpoint = args.checkpoint or args.initialize_from
    if source_checkpoint is None:
        model = get_peft_model(model, LoraConfig(
            r=64,
            lora_alpha=128,
            lora_dropout=.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        ))
    else:
        model = PeftModel.from_pretrained(model, source_checkpoint, is_trainable=True)
        load_visual_merger(model, source_checkpoint)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.bfloat16)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    config = _find_config(model)
    heads = int(_value(_value(config, "text_config", config), "num_attention_heads"))
    selector = HeadSelector(heads)
    if source_checkpoint is not None:
        head = torch.load(source_checkpoint / "selection_head.pt", map_location="cpu", weights_only=False)
        selector.layer_head_weights.data.copy_(head["layer_head_weights"])
    optimizer = torch.optim.AdamW([
        {"params": [parameter for parameter in model.parameters() if parameter.requires_grad], "lr": learning_rate},
        {"params": selector.parameters(), "lr": 1e-4},
    ], weight_decay=0.0)
    scheduler = scheduler_for(optimizer, 10, scheduler_steps)
    model, selector, optimizer, pair_loader, replay_loader = accelerator.prepare(
        model, selector, optimizer, pair_loader, replay_loader
    )
    model.train()
    selector.train()
    iterators = [iter(pair_loader), iter(replay_loader)]
    completed, micro_step = (
        restore(args.checkpoint, optimizer, scheduler, accelerator, args.stage, 64)
        if args.checkpoint
        else (0, 0)
    )
    if micro_step:
        for skipped in range(micro_step):
            index = 0 if skipped % pair_every == pair_every - 1 else 1
            _, iterators[index] = next_row((pair_loader, replay_loader)[index], iterators[index])
    target = args.steps
    optimizer.zero_grad(set_to_none=True)
    while completed < target:
        active_loaders, active_iterators = (pair_loader, replay_loader), iterators
        competitor_paired = micro_step % pair_every == pair_every - 1
        index = 0 if competitor_paired else 1
        row, active_iterators[index] = next_row(active_loaders[index], active_iterators[index])
        with accelerator.accumulate(model, selector):
            inputs, labels, query = encode(processor, row, accelerator.device)
            visual = torch.nonzero(inputs["input_ids"][0] == int(_value(config, "image_token_id")), as_tuple=False).flatten()
            keep = int(labels.ne(-100).sum()) + 1
            context = Attention(model, query, visual) if competitor_paired else nullcontext()
            with context as attention:
                output = model(**inputs, use_cache=False, logits_to_keep=keep)
                if competitor_paired:
                    scores = selector(attention.ordered())
                    selection_term = selection_loss(
                        scores,
                        row,
                        inputs["image_grid_thw"][0],
                        config,
                        margin=.3,
                        pair_weight=.5,
                    )
                else:
                    selection_term = output.logits.sum() * 0
                coord_loss = coordinate_loss(output.logits, inputs["input_ids"], labels)
                loss = coord_loss + auxiliary_weight * selection_term
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
                print(f"step={completed} loss={float(loss):.4f} coord={float(coord_loss):.4f} selection={float(selection_term):.4f}", flush=True)
    save(
        accelerator,
        model,
        selector,
        optimizer,
        scheduler,
        processor,
        args.output,
        completed,
        args.stage,
        micro_step,
    )
    if accelerator.is_main_process:
        run_config = vars(args) | {
            "base_model": BASE_MODEL,
            "base_revision": BASE_REVISION,
        }
        (args.output / "run_config.json").write_text(
            json.dumps(run_config, default=str, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
