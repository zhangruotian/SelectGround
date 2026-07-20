from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.cache_utils import DynamicCache


PROMPT = """You are an expert GUI grounding model.
Given a screenshot and an instruction, point to the UI element that should be clicked.
Return only one point as [x, y], where x and y are normalized integers from 0 to 1000 relative to the full image.
For an element with area, return the center point.
Instruction: {instruction}"""


class SelectGround:
    """SelectGround direct grounding and ConGround test-time inference."""

    def __init__(self, checkpoint: str = "ruotian/SelectGround-8B") -> None:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            checkpoint_path = Path(snapshot_download(checkpoint))
        adapter_config = json.loads((checkpoint_path / "adapter_config.json").read_text())
        base_model = adapter_config["base_model_name_or_path"]
        revision = adapter_config["revision"]
        self.agreement_radius = 12.0 if "Qwen3-VL-8B" in base_model else 1.0
        model = AutoModelForImageTextToText.from_pretrained(
            base_model,
            revision=revision,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        self.model = PeftModel.from_pretrained(model, checkpoint_path)
        merger = torch.load(checkpoint_path / "visual_merger.pt", map_location="cpu")
        parameters = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, value in merger.get("state_dict", merger).items():
                parameters[name].copy_(value.to(parameters[name].device, parameters[name].dtype))
        self.model.eval()
        self.model.config.use_cache = True
        self.processor = AutoProcessor.from_pretrained(
            base_model,
            revision=revision,
            min_pixels=3136,
            max_pixels=8847360,
        )
        self.device = next(self.model.parameters()).device
        self.core = _find_core(self.model)
        config = _find_config(self.model)
        self.image_token_id = int(_value(config, "image_token_id"))
        self.merge_size = int(_value(_value(config, "vision_config"), "spatial_merge_size", 2))

    def predict(self, image: str | Path | Image.Image, instruction: str, *, conground: bool = False) -> dict[str, Any]:
        source = Image.open(image).convert("RGB") if not isinstance(image, Image.Image) else image.convert("RGB")
        size = source.size
        if not conground:
            direct, _, _, _, _ = self._direct(source, size, instruction)
            return {"method": "SelectGround", **direct}

        p0, p1, grid = self._observe(source, size, instruction)
        q0, h0 = self._crop_observations(source, instruction, p0["point"])
        distance = _token_distance(p0["point"], p1["point"], size, grid)
        candidates = {"p0": p0, "p1": p1, "q0": q0, "h0": h0}
        if distance <= self.agreement_radius:
            q1, h1 = self._crop_observations(source, instruction, p1["point"])
            candidates.update(q1=q1, h1=h1)
            selected = _medoid(candidates, size, grid)
        else:
            selected = "q0"
        result = candidates[selected]
        return {
            "method": "SelectGround+ConGround",
            "point": result["point"],
            "normalized_point": result["normalized_point"],
            "raw_response": result["raw_response"],
            "selected": selected,
            "agreement": distance <= self.agreement_radius,
            "hypothesis_distance": distance,
            "candidates": {name: value["point"] for name, value in candidates.items()},
        }

    def _observe(
        self, image: Image.Image, logical_size: tuple[int, int], instruction: str
    ) -> tuple[dict[str, Any], dict[str, Any], tuple[int, int]]:
        direct, scores, cache, position_ids, state = self._direct(image, logical_size, instruction)
        point = direct["point"]
        current = _point_to_index(point, logical_size, state["grid"]) if point is not None else int(scores.argmax())
        incumbent, alternative = _hypotheses(scores, current, state["grid"])
        raw = self._reconsider(state, cache, position_ids, incumbent, alternative)
        return direct, _prediction(raw, logical_size), state["grid"]

    def _crop_observations(
        self, image: Image.Image, instruction: str, anchor: list[float] | None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if anchor is None:
            invalid = {"point": None, "normalized_point": None, "raw_response": ""}
            return invalid, invalid
        box = _crop_box(anchor, image.size)
        crop = image.crop(box)
        logical_size = crop.size
        zoomed = crop.resize((2 * logical_size[0], 2 * logical_size[1]), Image.Resampling.LANCZOS)
        direct, challenge, _ = self._observe(zoomed, logical_size, instruction)
        return _map_crop(direct, box, image.size), _map_crop(challenge, box, image.size)

    def _direct(
        self, image: Image.Image, logical_size: tuple[int, int], instruction: str
    ) -> tuple[dict[str, Any], torch.Tensor, DynamicCache, torch.Tensor, dict[str, Any]]:
        inputs = self._inputs(image, instruction)
        input_ids = inputs["input_ids"]
        length = int(input_ids.shape[1])
        visual = torch.nonzero(input_ids[0] == self.image_token_id, as_tuple=False).flatten()
        image_grid = inputs["image_grid_thw"][0].detach().cpu().long()
        grid = (int(image_grid[1]) // self.merge_size, int(image_grid[2]) // self.merge_size)
        if visual.numel() != grid[0] * grid[1]:
            raise RuntimeError("visual-token grid does not match the encoded image")
        position_ids, _ = self.core.get_rope_index(
            input_ids,
            inputs.get("image_grid_thw"),
            inputs.get("video_grid_thw"),
            attention_mask=inputs.get("attention_mask"),
        )
        cache = DynamicCache(config=self.core.language_model.config)
        attention = _Attention(self.model, length - 1, visual, list(range(18, 24)))
        with torch.inference_mode(), attention:
            output = self.model(
                **inputs,
                past_key_values=cache,
                position_ids=position_ids,
                cache_position=torch.arange(length, device=self.device),
                use_cache=True,
                logits_to_keep=1,
            )
        scores = torch.stack(attention.ordered()).float().cpu().mean(dim=(0, 1))
        raw = self._decode(output.logits[:, -1, :], _clone_cache(cache), position_ids[:, :, -1:] + 1)
        state = {"inputs": inputs, "visual": visual, "length": length, "grid": grid}
        return _prediction(raw, logical_size), scores, cache, position_ids, state

    def _reconsider(
        self,
        state: dict[str, Any],
        first_cache: DynamicCache,
        position_ids: torch.Tensor,
        incumbent: torch.Tensor,
        alternative: torch.Tensor,
    ) -> str:
        inputs, visual, length = state["inputs"], state["visual"], state["length"]
        explore, reconcile = _visibilities(length, visual, incumbent, alternative)
        second_inputs = {
            key: value
            for key, value in inputs.items()
            if key in {"input_ids", "pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"}
        }
        with torch.inference_mode(), _Prefix(self.model, explore, reconcile):
            output = self.model(
                **second_inputs,
                past_key_values=_clone_cache(first_cache),
                attention_mask=torch.ones((1, 2 * length), dtype=torch.long, device=self.device),
                position_ids=position_ids,
                cache_position=torch.arange(length, 2 * length, device=self.device),
                use_cache=True,
                logits_to_keep=1,
            )
        keep = torch.cat((torch.arange(length), visual.detach().cpu() + length))
        layers = []
        for keys, values in output.past_key_values.to_legacy_cache():
            indices = keep.to(keys.device)
            layers.append((keys.index_select(2, indices), values.index_select(2, indices)))
        cache = DynamicCache.from_legacy_cache(tuple(layers))
        return self._decode(output.logits[:, -1, :], cache, position_ids[:, :, -1:] + 1)

    @torch.inference_mode()
    def _decode(self, logits: torch.Tensor, cache: DynamicCache, position_ids: torch.Tensor) -> str:
        stop = {
            int(token)
            for token in (self.processor.tokenizer.eos_token_id, self.processor.tokenizer.pad_token_id)
            if token is not None
        }
        generated = []
        for _ in range(32):
            token = logits.argmax(dim=-1)
            generated.append(int(token))
            if generated[-1] in stop:
                break
            output = self.model(
                input_ids=token.view(1, 1),
                past_key_values=cache,
                position_ids=position_ids,
                cache_position=torch.tensor([cache.get_seq_length()], device=token.device),
                use_cache=True,
                logits_to_keep=1,
            )
            cache = output.past_key_values
            logits = output.logits[:, -1, :]
            position_ids = position_ids + 1
        return self.processor.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()

    def _inputs(self, image: Image.Image, instruction: str) -> Any:
        from qwen_vl_utils import process_vision_info

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT.format(instruction=instruction)},
            ],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        kwargs = {"text": [text], "images": images, "padding": True, "return_tensors": "pt"}
        if videos is not None:
            kwargs["videos"] = videos
        return self.processor(**kwargs).to(self.device)


class _Attention:
    def __init__(self, model: Any, query: int, visual: torch.Tensor, layers: list[int]) -> None:
        self.model, self.query, self.visual, self.layers = model, query, visual, layers
        self.handles: list[Any] = []
        self.values: dict[int, torch.Tensor] = {}

    def __enter__(self) -> "_Attention":
        wanted = set(self.layers)
        for module in self.model.modules():
            layer = getattr(module, "layer_idx", None)
            if layer in wanted and hasattr(module, "q_proj"):
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
                ).detach()

        return hook

    def ordered(self) -> list[torch.Tensor]:
        if set(self.values) != set(self.layers):
            raise RuntimeError("failed to read all six attention layers")
        return [self.values[layer] for layer in self.layers]


class _Prefix:
    def __init__(self, model: Any, explore: torch.Tensor, reconcile: torch.Tensor) -> None:
        self.model, self.explore, self.reconcile = model, explore, reconcile
        self.handles: list[Any] = []

    def __enter__(self) -> "_Prefix":
        for module in self.model.modules():
            layer = getattr(getattr(module, "self_attn", None), "layer_idx", None)
            if layer is not None and int(layer) >= 2:
                self.handles.append(module.register_forward_pre_hook(self._hook(int(layer)), with_kwargs=True))
        return self

    def __exit__(self, *_: Any) -> None:
        for handle in self.handles:
            handle.remove()

    def _hook(self, layer: int):
        def hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
            mask = kwargs["attention_mask"].clone()
            visible = self.explore if layer < 12 else self.reconcile
            hidden = torch.nonzero(~visible, as_tuple=False).flatten().to(mask.device)
            if mask.dtype == torch.bool:
                mask[..., hidden] = False
            else:
                mask[..., hidden] = torch.finfo(mask.dtype).min
            return args, {**kwargs, "attention_mask": mask}

        return hook


def _attention_logits(
    attention: Any,
    hidden: torch.Tensor,
    query_position: int,
    visual_positions: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

    shape = (*hidden.shape[:-1], -1, int(attention.head_dim))
    queries = attention.q_norm(attention.q_proj(hidden).view(shape)).transpose(1, 2)
    keys = attention.k_norm(attention.k_proj(hidden).view(shape)).transpose(1, 2)
    queries, keys = apply_rotary_pos_emb(queries, keys, *position_embeddings)
    groups = int(getattr(attention, "num_key_value_groups", 1))
    if groups > 1:
        batch, heads, length, dim = keys.shape
        keys = keys[:, :, None, :, :].expand(batch, heads, groups, length, dim).reshape(
            batch, heads * groups, length, dim
        )
    visual = keys.index_select(2, visual_positions.to(keys.device))
    return ((queries[:, :, query_position, :].unsqueeze(2) * visual).sum(-1) * attention.scaling)[0]


def _hypotheses(
    scores: torch.Tensor, current: int, grid: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = grid
    rows = torch.arange(height).repeat_interleave(width)
    columns = torch.arange(width).repeat(height)
    current_row, current_column = divmod(current, width)
    distance = (rows - current_row).square() + (columns - current_column).square()
    order = torch.argsort(distance * (height * width + 1) + torch.arange(height * width))
    incumbent = order[:64]
    occupied = torch.zeros(height * width, dtype=torch.bool)
    occupied[incumbent] = True
    far = torch.maximum((rows - current_row).abs(), (columns - current_column).abs()) >= 11
    eligible = ~occupied & far
    if not eligible.any():
        eligible = ~occupied
    alternative_center = int(scores.masked_fill(~eligible, -torch.inf).argmax())
    row, column = divmod(alternative_center, width)
    distance = (rows - row).square() + (columns - column).square()
    order = torch.argsort(distance * (height * width + 1) + torch.arange(height * width))
    alternative = order[~occupied[order]][:64]
    return incumbent.sort().values, alternative.sort().values


def _visibilities(
    length: int, visual: torch.Tensor, incumbent: torch.Tensor, alternative: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    visual = visual.detach().cpu()
    explore = torch.zeros(length, dtype=torch.bool)
    explore[visual[alternative]] = True
    explore[int(visual[-1]) + 1 :] = True
    reconcile = explore.clone()
    reconcile[visual[incumbent]] = True
    return explore, reconcile


def _prediction(raw: str, size: tuple[int, int]) -> dict[str, Any]:
    match = re.search(r"[\[(]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\])]", raw)
    normalized = [float(match.group(1)), float(match.group(2))] if match else None
    point = [normalized[0] / 1000 * size[0], normalized[1] / 1000 * size[1]] if normalized else None
    return {"point": point, "normalized_point": normalized, "raw_response": raw}


def _map_crop(prediction: dict[str, Any], box: tuple[int, int, int, int], size: tuple[int, int]) -> dict[str, Any]:
    point = prediction["point"]
    mapped = [box[0] + point[0], box[1] + point[1]] if point is not None else None
    normalized = [mapped[0] / size[0] * 1000, mapped[1] / size[1] * 1000] if mapped else None
    raw = f"[{round(normalized[0])},{round(normalized[1])}]" if normalized else ""
    return {"point": mapped, "normalized_point": normalized, "raw_response": raw}


def _crop_box(point: list[float], size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    crop_width, crop_height = min(width, max(320, round(.4 * width))), min(height, max(320, round(.4 * height)))
    left = round(min(max(0.0, point[0] - crop_width / 2), width - crop_width))
    top = round(min(max(0.0, point[1] - crop_height / 2), height - crop_height))
    return left, top, left + crop_width, top + crop_height


def _point_to_index(point: list[float], size: tuple[int, int], grid: tuple[int, int]) -> int:
    x = min(max(point[0], 0.0), math.nextafter(float(size[0]), 0.0))
    y = min(max(point[1], 0.0), math.nextafter(float(size[1]), 0.0))
    return min(int(y / size[1] * grid[0]), grid[0] - 1) * grid[1] + min(int(x / size[0] * grid[1]), grid[1] - 1)


def _token_distance(
    first: list[float] | None,
    second: list[float] | None,
    size: tuple[int, int],
    grid: tuple[int, int],
) -> float:
    if first is None or second is None:
        return math.inf
    dx = (first[0] - second[0]) / size[0] * grid[1]
    dy = (first[1] - second[1]) / size[1] * grid[0]
    return math.hypot(dx, dy)


def _medoid(candidates: dict[str, dict[str, Any]], size: tuple[int, int], grid: tuple[int, int]) -> str:
    priority = {name: index for index, name in enumerate(("q0", "q1", "h0", "h1", "p0", "p1"))}
    valid = {name: value["point"] for name, value in candidates.items() if value["point"] is not None}
    def cost(name: str) -> tuple[float, int]:
        point = valid[name]
        total = sum(_token_distance(point, other, size, grid) for other in valid.values())
        return total, priority[name]
    return min(valid, key=cost)


def _clone_cache(cache: DynamicCache) -> DynamicCache:
    return DynamicCache.from_legacy_cache(tuple((keys.clone(), values.clone()) for keys, values in cache.to_legacy_cache()))


def _find_core(model: Any) -> Any:
    for obj in _model_objects(model):
        if hasattr(obj, "get_rope_index"):
            return obj
    raise RuntimeError("Qwen core was not found")


def _find_config(model: Any) -> Any:
    for obj in _model_objects(model):
        config = getattr(obj, "config", None)
        if config is not None and _value(config, "image_token_id") is not None:
            return config
    raise RuntimeError("Qwen vision config was not found")


def _model_objects(model: Any):
    seen, stack = set(), [model]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        yield obj
        stack.extend(child for name in ("module", "base_model", "model") if (child := getattr(obj, name, None)) is not None)


def _value(config: Any, name: str, default: Any = None) -> Any:
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)
