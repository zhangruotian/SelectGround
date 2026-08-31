"""The fixed two-stage SelectGround-8B recipe."""
import argparse
import json
import os
import random
import math
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
from transformers import AutoModelForImageTextToText, AutoProcessor, get_scheduler
from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv

from selectground import BASE_MODEL, BASE_REVISION, PROMPT

LAYERS = list(range(18, 24))
SEED = 20260625


def _attention_logits(
    attention: Any,
    hidden_states: torch.Tensor,
    query_position: int,
    visual_positions: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
) -> torch.Tensor:
    head_dim = int(attention.head_dim)
    hidden_shape = (*hidden_states.shape[:-1], -1, head_dim)
    query_states = attention.q_norm(attention.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = attention.k_norm(attention.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, *position_embeddings)
    key_states = repeat_kv(key_states, attention.num_key_value_groups)
    positions = visual_positions.to(device=hidden_states.device, dtype=torch.long)
    query = query_states[:, :, int(query_position), :]
    visual_keys = key_states.index_select(2, positions)
    logits = (query.unsqueeze(2) * visual_keys).sum(dim=-1) * attention.scaling
    return logits.squeeze(0)


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
            self.values[layer] = _attention_logits(
                module, kwargs["hidden_states"], self.query, self.visual, kwargs["position_embeddings"]
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
    vision = config.vision_config
    patch, merge = vision.patch_size, vision.spatial_merge_size
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


def save(accelerator, model, selector, optimizer, scheduler, output, step, micro_step):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(output, safe_serialization=True)
        path = output / "adapter_config.json"
        config = json.loads(path.read_text())
        config["revision"] = BASE_REVISION
        path.write_text(json.dumps(config, indent=2) + "\n")
        torch.save({
            "layers": LAYERS,
            "layer_head_weights": accelerator.unwrap_model(selector).layer_head_weights.detach().cpu(),
        }, output / "selection_head.pt")
        torch.save({
            "completed": step, "micro_step": micro_step,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        }, output / "training_state.pt")
    accelerator.wait_for_everyone()
    torch.save({
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
    }, output / f"rng_state_rank_{accelerator.process_index}.pt")
    accelerator.wait_for_everyone()


def restore(path, optimizer, scheduler, rank):
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    rng = torch.load(path / f"rng_state_rank_{rank}.pt", map_location="cpu", weights_only=False)
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    torch.cuda.set_rng_state_all(rng["cuda"])
    return state["completed"], state["micro_step"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("main", "refinement"), required=True)
    parser.add_argument("--steps", type=int, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--resume", type=Path)
    source.add_argument("--initialize-from", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output directory; use --resume to load a checkpoint.")
    if args.stage == "refinement" and not (args.resume or args.initialize_from):
        parser.error("refinement requires --initialize-from or --resume")
    rank = int(os.environ["RANK"])
    set_seed(SEED + rank)
    refinement = args.stage == "refinement"
    pair_every, learning_rate, horizon, aux_weight = (
        (3, 1e-6, 30, .05) if refinement else (2, 3.45e-5, 320, .1)
    )
    pairs = read_rows(args.data / "data/refinement_pairs.jsonl")
    replay = read_rows(args.data / "data/refinement_replay.jsonl")
    if not refinement:
        pairs = read_rows(args.data / "data/train_pairs.jsonl") + pairs
        replay = read_rows(args.data / "data/train_replay.jsonl") + replay
    offset = 3000 if refinement else 1000
    pair_loader = loader(pairs, args.data, SEED + offset + 1)
    replay_loader = loader(replay, args.data, SEED + offset + 1001)
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, min_pixels=3136, max_pixels=8847360
    )
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    checkpoint = args.resume or args.initialize_from
    if checkpoint:
        model = PeftModel.from_pretrained(model, checkpoint, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=64, lora_alpha=128, lora_dropout=.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        ))
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.bfloat16)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    config = model.config
    selector = HeadSelector(config.text_config.num_attention_heads)
    if checkpoint:
        state = torch.load(checkpoint / "selection_head.pt", map_location="cpu", weights_only=True)
        selector.layer_head_weights.data.copy_(state["layer_head_weights"])
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": learning_rate},
        {"params": selector.parameters(), "lr": 1e-4},
    ], weight_decay=0.0)
    scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=10, num_training_steps=horizon)
    # Initialize distributed training after loading the replicated adapter.
    accelerator = Accelerator(
        gradient_accumulation_steps=64,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )
    if accelerator.num_processes != 2:
        raise ValueError("This recipe uses exactly two GPU processes (effective batch 128).")
    model, selector, optimizer, pair_loader, replay_loader = accelerator.prepare(
        model, selector, optimizer, pair_loader, replay_loader
    )
    model.train()
    selector.train()
    iterators = [iter(pair_loader), iter(replay_loader)]
    completed, micro_step = restore(args.resume, optimizer, scheduler, rank) if args.resume else (0, 0)
    for skipped in range(micro_step):
        index = 0 if skipped % pair_every == pair_every - 1 else 1
        _, iterators[index] = next_row((pair_loader, replay_loader)[index], iterators[index])
    optimizer.zero_grad(set_to_none=True)
    while completed < args.steps:
        paired = micro_step % pair_every == pair_every - 1
        index = 0 if paired else 1
        row, iterators[index] = next_row((pair_loader, replay_loader)[index], iterators[index])
        with accelerator.accumulate(model, selector):
            inputs, labels, query = encode(processor, row, accelerator.device)
            visual = torch.nonzero(inputs["input_ids"][0] == config.image_token_id, as_tuple=False).flatten()
            keep = int(labels.ne(-100).sum()) + 1
            with Attention(model, query, visual) if paired else nullcontext() as attention:
                output = model(**inputs, use_cache=False, logits_to_keep=keep)
                aux = selection_loss(
                    selector(attention.ordered()), row, inputs["image_grid_thw"][0],
                    config, margin=.3, pair_weight=.5,
                ) if paired else output.logits.sum() * 0
                coord = coordinate_loss(output.logits, inputs["input_ids"], labels)
                loss = coord + aux_weight * aux
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
                print(f"step={completed} loss={loss.item():.4f} coord={coord.item():.4f} selection={aux.item():.4f}", flush=True)
            if completed % 10 == 0 or completed == args.steps:
                save(accelerator, model, selector, optimizer, scheduler, args.output, completed, micro_step)
    accelerator.end_training()


if __name__ == "__main__":
    main()
