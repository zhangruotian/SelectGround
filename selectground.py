import json
import re
from contextlib import nullcontext
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
    """SelectGround direct grounding and CVL test-time inference."""

    def __init__(self, checkpoint: str = "ruotian/SelectGround-8B") -> None:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            checkpoint_path = Path(snapshot_download(checkpoint))
        adapter_config = json.loads((checkpoint_path / "adapter_config.json").read_text())
        base_model = adapter_config["base_model_name_or_path"]
        revision = adapter_config["revision"]
        self.cvl_proximity_weight = 0.0 if "30B" in base_model else 0.5
        self.cvl_q0_weight = 0.5 if "30B" in base_model else 0.0
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
        base = self.model.base_model.model
        self.core = base.model
        self.image_token_id = int(base.config.image_token_id)
        self.merge_size = int(base.config.vision_config.spatial_merge_size)

    def predict(
        self,
        image: str | Path | Image.Image,
        instruction: str,
        *,
        cvl: bool = False,
    ) -> dict[str, Any]:
        source = Image.open(image).convert("RGB") if not isinstance(image, Image.Image) else image.convert("RGB")
        size = source.size
        p0, full_view, raw_scores, grid = self._observe(
            source,
            size,
            instruction,
            box=(0, 0, size[0], size[1]),
            capture_attention=cvl,
        )
        if not cvl:
            return {"method": "SelectGround", **p0}

        if p0["point"] is None:
            row, column = divmod(int(raw_scores.argmax()), grid[1])
            p0 = _point_prediction(
                [(column + 0.5) / grid[1] * size[0], (row + 0.5) / grid[0] * size[1]],
                size,
            )
        current = _point_to_index(p0["point"], size, grid)
        p0_anchor = p0["point"]
        peaks = _top_peaks(raw_scores, current, size, grid)
        width, height = size
        proposals = [
            (p0_anchor, 0.40),
            ([0.3 * width, 0.3 * height], 0.60),
            ([0.7 * width, 0.3 * height], 0.60),
            ([0.3 * width, 0.7 * height], 0.60),
            ([0.7 * width, 0.7 * height], 0.60),
        ]
        observations = [
            self._crop_view(source, instruction, anchor, fraction)
            for anchor, fraction in proposals
        ]
        q0 = observations[0][0]
        grids = [observation[0] for observation in observations[1:]]
        top_predictions = [_point_prediction(point, size) for point in peaks]
        predictions = [p0, q0, *top_predictions, *grids]
        points = [prediction["point"] for prediction in predictions]
        source_views = [0, 1, 0, 0, 0, 2, 3, 4, 5]
        selected = self._cvl_select(
            points,
            [full_view, *(view for _, view in observations)],
            source_views,
            size,
        )
        result = predictions[selected]
        return {
            "method": "SelectGround+CVL",
            "point": result["point"],
            "normalized_point": result["normalized_point"],
            "raw_response": result["raw_response"],
        }

    def _crop_view(
        self, image: Image.Image, instruction: str, anchor: list[float], fraction: float
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        box = _crop_box(anchor, image.size, fraction)
        crop = image.crop(box)
        logical_size = crop.size
        zoomed = crop.resize((2 * logical_size[0], 2 * logical_size[1]), Image.Resampling.LANCZOS)
        direct, view, _, _ = self._observe(
            zoomed,
            logical_size,
            instruction,
            box=box,
            capture_attention=False,
        )
        mapped = _map_crop(direct, box, image.size)
        return (mapped if mapped["point"] is not None else _point_prediction(anchor, image.size)), view

    def _observe(
        self,
        image: Image.Image,
        logical_size: tuple[int, int],
        instruction: str,
        *,
        box: tuple[int, int, int, int],
        capture_attention: bool,
    ) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor | None, tuple[int, int]]:
        inputs = self._inputs(image, instruction)
        input_ids = inputs["input_ids"]
        length = int(input_ids.shape[1])
        visual = torch.nonzero(input_ids[0] == self.image_token_id, as_tuple=False).flatten()
        image_grid = inputs["image_grid_thw"][0].detach().cpu().long()
        grid = (int(image_grid[1]) // self.merge_size, int(image_grid[2]) // self.merge_size)
        position_ids, _ = self.core.get_rope_index(
            input_ids,
            inputs.get("image_grid_thw"),
            inputs.get("video_grid_thw"),
            attention_mask=inputs.get("attention_mask"),
        )
        cache = DynamicCache(config=self.core.language_model.config)
        attention = (
            _Attention(self.model, length - 1, visual)
            if capture_attention
            else nullcontext()
        )
        with torch.inference_mode(), attention:
            output = self.model(
                **inputs,
                past_key_values=cache,
                position_ids=position_ids,
                cache_position=torch.arange(length, device=self.device),
                use_cache=True,
                logits_to_keep=1,
            )
        raw_scores = None
        if capture_attention:
            layer_scores = torch.stack(
                [value.float().cpu() for value in attention.ordered()]
            )
            raw_scores = layer_scores.mean(dim=(0, 1))
        raw = self._decode(output.logits[:, -1, :], _clone_cache(cache), position_ids[:, :, -1:] + 1)
        view = {
            "box": box,
            "logical_size": logical_size,
            "initial_logits": output.logits[:, -1, :],
            "cache": cache,
            "position_ids": position_ids,
        }
        return _prediction(raw, logical_size), view, raw_scores, grid

    @torch.inference_mode()
    def _score_responses(
        self, view: dict[str, Any], responses: list[str]
    ) -> dict[str, float]:
        unique = list(dict.fromkeys(responses))
        token_ids = [
            self.processor.tokenizer(response, add_special_tokens=False)["input_ids"]
            for response in unique
        ]
        lengths = [len(values) for values in token_ids]
        initial = torch.log_softmax(view["initial_logits"].detach().float(), dim=-1)[0]
        logprobs = [[float(initial[values[0]])] for values in token_ids]
        max_length = max(lengths)
        if max_length > 1:
            tokenizer = self.processor.tokenizer
            batch = len(unique)
            continuation = torch.full(
                (batch, max_length - 1),
                tokenizer.pad_token_id,
                dtype=torch.long,
                device=self.device,
            )
            mask = torch.zeros_like(continuation, dtype=torch.bool)
            for index, values in enumerate(token_ids):
                prefix = values[:-1]
                continuation[index, : len(prefix)] = torch.tensor(
                    prefix, device=self.device
                )
                mask[index, : len(prefix)] = True
            cache = _clone_cache(view["cache"])
            cache.batch_repeat_interleave(batch)
            prompt_length = view["cache"].get_seq_length()
            attention_mask = torch.cat(
                (
                    torch.ones(
                        (batch, prompt_length),
                        dtype=torch.long,
                        device=self.device,
                    ),
                    mask.long(),
                ),
                dim=1,
            )
            offsets = torch.arange(max_length - 1, device=self.device).view(1, 1, -1)
            position_ids = (
                view["position_ids"][:, :, -1:] + 1 + offsets
            ).expand(-1, batch, -1)
            output = self.model(
                input_ids=continuation,
                past_key_values=cache,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=torch.arange(
                    prompt_length,
                    prompt_length + max_length - 1,
                    device=self.device,
                ),
                use_cache=True,
                logits_to_keep=max_length - 1,
            )
            for index, values in enumerate(token_ids):
                if len(values) == 1:
                    continue
                logits = output.logits[index, : len(values) - 1].detach().float()
                labels = torch.tensor(values[1:], device=logits.device)
                selected = torch.log_softmax(logits, dim=-1).gather(
                    1, labels[:, None]
                )[:, 0]
                logprobs[index].extend(float(value) for value in selected.cpu())
        return {
            response: sum(values) / len(values)
            for response, values in zip(unique, logprobs, strict=True)
        }

    def _cvl_select(
        self,
        points: list[list[float]],
        views: list[dict[str, Any]],
        source_views: list[int],
        size: tuple[int, int],
    ) -> int:
        per_view = []
        for view in views:
            box = view["box"]
            local = [
                (
                    [point[0] - box[0], point[1] - box[1]]
                    if box[0] <= point[0] < box[2] and box[1] <= point[1] < box[3]
                    else None
                )
                for point in points
            ]
            responses = [
                _point_response(point, view["logical_size"])
                for point in local
                if point is not None
            ]
            scores = self._score_responses(view, responses)
            unique = torch.tensor(list(scores.values()), dtype=torch.float32)
            normalized = {
                response: float(value)
                for response, value in zip(
                    scores,
                    _zscore(unique).tolist(),
                    strict=True,
                )
            }
            per_view.append(
                [
                    normalized[_point_response(point, view["logical_size"])]
                    if point is not None
                    else None
                    for point in local
                ]
            )

        cross = []
        for candidate, source_view in enumerate(source_views):
            evidence = [
                values[candidate]
                for view_index, values in enumerate(per_view)
                if view_index != source_view and values[candidate] is not None
            ]
            cross.append(sum(evidence) / len(evidence))
        cross_score = _zscore(torch.tensor(cross, dtype=torch.float32))
        normalized_points = torch.tensor(
            [[point[0] / size[0], point[1] / size[1]] for point in points],
            dtype=torch.float32,
        )
        proximity = _zscore(-torch.cdist(normalized_points, normalized_points)[:, 1])
        score = cross_score + self.cvl_proximity_weight * proximity
        score[1] += self.cvl_q0_weight
        return int(score.argmax())

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
        images, _ = process_vision_info(messages)
        return self.processor(
            text=[text], images=images, padding=True, return_tensors="pt"
        ).to(self.device)


class _Attention:
    def __init__(self, model: Any, query: int, visual: torch.Tensor) -> None:
        self.model, self.query, self.visual = model, query, visual
        self.layers = range(18, 24)
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


def _prediction(raw: str, size: tuple[int, int]) -> dict[str, Any]:
    match = re.search(r"[\[(]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\])]", raw)
    normalized = [float(match.group(1)), float(match.group(2))] if match else None
    point = [normalized[0] / 1000 * size[0], normalized[1] / 1000 * size[1]] if normalized else None
    return {"point": point, "normalized_point": normalized, "raw_response": raw}


def _point_prediction(point: list[float], size: tuple[int, int]) -> dict[str, Any]:
    normalized = [point[0] / size[0] * 1000, point[1] / size[1] * 1000]
    return {
        "point": point,
        "normalized_point": normalized,
        "raw_response": f"[{round(normalized[0])},{round(normalized[1])}]",
    }


def _point_response(point: list[float], size: tuple[int, int]) -> str:
    return f"[{round(point[0] / size[0] * 1000)},{round(point[1] / size[1] * 1000)}]"


def _map_crop(prediction: dict[str, Any], box: tuple[int, int, int, int], size: tuple[int, int]) -> dict[str, Any]:
    point = prediction["point"]
    mapped = [box[0] + point[0], box[1] + point[1]] if point is not None else None
    normalized = [mapped[0] / size[0] * 1000, mapped[1] / size[1] * 1000] if mapped else None
    raw = f"[{round(normalized[0])},{round(normalized[1])}]" if normalized else ""
    return {"point": mapped, "normalized_point": normalized, "raw_response": raw}


def _crop_box(
    point: list[float], size: tuple[int, int], fraction: float
) -> tuple[int, int, int, int]:
    width, height = size
    crop_width = min(width, max(320, round(fraction * width)))
    crop_height = min(height, max(320, round(fraction * height)))
    left = round(min(max(0.0, point[0] - crop_width / 2), width - crop_width))
    top = round(min(max(0.0, point[1] - crop_height / 2), height - crop_height))
    return left, top, left + crop_width, top + crop_height


def _point_to_index(point: list[float], size: tuple[int, int], grid: tuple[int, int]) -> int:
    row = min(int(point[1] / size[1] * grid[0]), grid[0] - 1)
    column = min(int(point[0] / size[0] * grid[1]), grid[1] - 1)
    return row * grid[1] + column


def _top_peaks(
    scores: torch.Tensor,
    current: int,
    size: tuple[int, int],
    grid: tuple[int, int],
) -> list[list[float]]:
    values = scores.float().view(*grid)
    pooled = torch.nn.functional.max_pool2d(values[None, None], 3, stride=1, padding=1)[0, 0]
    rows, columns = torch.meshgrid(
        torch.arange(grid[0]), torch.arange(grid[1]), indexing="ij"
    )
    current_row, current_column = divmod(current, grid[1])
    eligible = torch.maximum((rows - current_row).abs(), (columns - current_column).abs()) >= 4
    available = eligible & (values >= pooled)
    peaks = []
    for _ in range(3):
        index = int(values.masked_fill(~available, -torch.inf).argmax())
        row, column = divmod(index, grid[1])
        peaks.append((row, column))
        available[max(0, row - 8) : row + 9, max(0, column - 8) : column + 9] = False
    return [
        [(column + 0.5) / grid[1] * size[0], (row + 0.5) / grid[0] * size[1]]
        for row, column in peaks
    ]


def _zscore(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)


def _clone_cache(cache: DynamicCache) -> DynamicCache:
    return DynamicCache.from_legacy_cache(tuple((keys.clone(), values.clone()) for keys, values in cache.to_legacy_cache()))
