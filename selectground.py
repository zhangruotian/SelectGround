import json
import math
import re
from itertools import combinations
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

LCR_PROMPT = """You are an expert UI element locator. Given a GUI image and a user's element description, provide the coordinates of the specified element as a single (x,y) point. The image resolution is height {height} and width {width}. For elements with area, return the center point.

Output the coordinate pair exactly:
(x,y)"""


class SelectGround:
    """SelectGround direct grounding and LCR test-time inference."""

    def __init__(self, checkpoint: str = "ruotian/SelectGround-8B") -> None:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            checkpoint_path = Path(snapshot_download(checkpoint))
        adapter_config = json.loads((checkpoint_path / "adapter_config.json").read_text())
        base_model = adapter_config["base_model_name_or_path"]
        revision = adapter_config["revision"]
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
        self.core = self.model.get_base_model().model
        self.vision_start_token_id = int(self.core.config.vision_start_token_id)
        self.vision_end_token_id = int(self.core.config.vision_end_token_id)
        self.merge_size = int(self.core.config.vision_config.spatial_merge_size)
        self.comma_token_id = int(
            self.processor.tokenizer(",", add_special_tokens=False)["input_ids"][0]
        )

    def predict(
        self,
        image: str | Path | Image.Image,
        instruction: str,
        *,
        lcr: bool = False,
        benchmark: str | None = None,
    ) -> dict[str, Any]:
        """Return a source-image click; benchmark only selects UI-Vision's crop size."""
        source = Image.open(image).convert("RGB") if not isinstance(image, Image.Image) else image.convert("RGB")
        size = source.size
        p0, attention, grid = self._observe(
            source, instruction, capture_attention=lcr, lcr_prompt=lcr
        )
        if not lcr:
            return {"method": "SelectGround", **p0}

        views = [
            *((box, 2.0) for box in _attention_crops(attention, grid, size, benchmark)),
            (_pixel_budget_crop(p0["point"], size, 501_760), 1.5),
        ]
        observations = [(p0, (0, 0, size[0], size[1]))]
        for box, scale in views:
            crop = source.crop(box)
            view = crop.resize(
                (round(crop.width * scale), round(crop.height * scale)),
                Image.Resampling.BICUBIC,
            )
            prediction, _, _ = self._observe(view, instruction, lcr_prompt=True)
            if prediction["point"] is not None:
                observations.append((_map_crop(prediction, box, size, scale), box))
        first, second = min(
            combinations(range(len(observations)), 2),
            key=lambda pair: (
                _distance(
                    observations[pair[0]][0]["point"],
                    observations[pair[1]][0]["point"],
                    size,
                ),
                pair,
            ),
        )
        selected = min(
            (first, second),
            key=lambda index: (_area(observations[index][1]), index),
        )
        result = observations[selected][0]
        return {
            "method": "SelectGround+LCR",
            "point": result["point"],
            "normalized_point": result["normalized_point"],
            "raw_response": result["raw_response"],
        }

    def _observe(
        self,
        image: Image.Image,
        instruction: str,
        *,
        capture_attention: bool = False,
        lcr_prompt: bool = False,
    ) -> tuple[dict[str, Any], torch.Tensor | None, tuple[int, int]]:
        inputs = self._inputs(image, instruction, lcr_prompt)
        input_ids = inputs["input_ids"]
        length = int(input_ids.shape[1])
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
            _Attention(
                self.core,
                input_ids,
                self.vision_start_token_id,
                self.vision_end_token_id,
            )
            if capture_attention
            else None
        )
        with torch.inference_mode():
            output = self.model(
                **inputs,
                past_key_values=cache,
                position_ids=position_ids,
                cache_position=torch.arange(length, device=self.device),
                use_cache=True,
                logits_to_keep=1,
            )
            raw = self._decode(
                output.logits[:, -1, :],
                cache,
                position_ids[:, :, -1:] + 1,
                attention,
            )
        return (
            _prediction(raw, image.size, integer=lcr_prompt),
            attention.scores if attention else None,
            grid,
        )

    def _decode(
        self,
        logits: torch.Tensor,
        cache: DynamicCache,
        position_ids: torch.Tensor,
        attention: "_Attention | None",
    ) -> str:
        stop = {
            token
            for token in (
                self.processor.tokenizer.eos_token_id,
                self.processor.tokenizer.pad_token_id,
            )
            if token is not None
        }
        generated = []
        for _ in range(32):
            token = int(logits.argmax())
            if token in stop:
                break
            generated.append(token)
            if attention is not None and token == self.comma_token_id:
                attention.install(cache)
            try:
                output = self.model(
                    input_ids=torch.tensor([[token]], device=self.device),
                    past_key_values=cache,
                    position_ids=position_ids,
                    cache_position=torch.tensor(
                        [cache.get_seq_length()], device=self.device
                    ),
                    use_cache=True,
                    logits_to_keep=1,
                )
            finally:
                if attention is not None and attention.handle is not None:
                    attention.remove()
            cache = output.past_key_values
            logits = output.logits[:, -1, :]
            position_ids = position_ids + 1
        return self.processor.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

    def _inputs(
        self, image: Image.Image, instruction: str, lcr_prompt: bool
    ) -> Any:
        from qwen_vl_utils import process_vision_info, smart_resize

        if lcr_prompt:
            resized_height, resized_width = smart_resize(
                image.height,
                image.width,
                factor=(
                    self.processor.image_processor.patch_size
                    * self.processor.image_processor.merge_size
                ),
                min_pixels=self.processor.image_processor.min_pixels,
                max_pixels=self.processor.image_processor.max_pixels,
            )
            image = image.resize((resized_width, resized_height))
            messages = [
                {
                    "role": "system",
                    "content": LCR_PROMPT.format(
                        height=resized_height, width=resized_width
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": instruction},
                    ],
                },
            ]
        else:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT.format(instruction=instruction)},
                ],
            }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        return self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)


class _Attention:
    def __init__(
        self,
        model: Any,
        input_ids: torch.Tensor,
        vision_start: int,
        vision_end: int,
    ) -> None:
        self.model = model
        self.handle: Any = None
        start = int(torch.nonzero(input_ids[0] == vision_start)[0]) + 1
        end = int(torch.nonzero(input_ids[0] == vision_end)[0])
        self.visual = torch.arange(start, end, device=input_ids.device)
        self.cache: DynamicCache | None = None
        self.scores: torch.Tensor | None = None

    def install(self, cache: DynamicCache) -> None:
        self.cache = cache
        layer = self.model.language_model.layers[24].self_attn
        self.handle = layer.register_forward_hook(self._hook, with_kwargs=True)

    def remove(self) -> None:
        self.handle.remove()
        self.handle = None

    def _hook(
        self,
        module: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            apply_rotary_pos_emb,
            repeat_kv,
        )

        hidden = kwargs.get("hidden_states", args[0] if args else None)
        shape = (*hidden.shape[:-1], -1, int(module.head_dim))
        query = module.q_norm(module.q_proj(hidden).view(shape)).transpose(1, 2)
        query, _ = apply_rotary_pos_emb(
            query, query, *kwargs["position_embeddings"]
        )
        keys = repeat_kv(
            self.cache.layers[module.layer_idx].keys,
            int(module.num_key_value_groups),
        )
        weights = torch.matmul(query, keys.transpose(-2, -1)) * module.scaling
        weights = weights.squeeze(2).softmax(-1).max(1).values[0]
        self.scores = (
            weights.index_select(0, self.visual.to(weights.device))
            .detach()
            .float()
            .cpu()
        )


def _prediction(
    raw: str, size: tuple[int, int], *, integer: bool = False
) -> dict[str, Any]:
    match = re.search(r"[\[(]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\])]", raw)
    normalized = [float(match.group(1)), float(match.group(2))] if match else None
    point = [normalized[0] / 1000 * size[0], normalized[1] / 1000 * size[1]] if normalized else None
    if point is not None and integer:
        point = [int(value) for value in point]
    return {"point": point, "normalized_point": normalized, "raw_response": raw}


def _map_crop(
    prediction: dict[str, Any],
    box: tuple[int, int, int, int],
    size: tuple[int, int],
    scale: float,
) -> dict[str, Any]:
    point = prediction["point"]
    mapped = [box[0] + point[0] / scale, box[1] + point[1] / scale]
    normalized = [mapped[0] / size[0] * 1000, mapped[1] / size[1] * 1000]
    raw = f"[{round(normalized[0])},{round(normalized[1])}]"
    return {"point": mapped, "normalized_point": normalized, "raw_response": raw}


def _pixel_budget_crop(
    point: list[float], size: tuple[int, int], pixels: int
) -> tuple[int, int, int, int]:
    width, height = size
    fraction = min(1.0, math.sqrt(pixels / (width * height)))
    crop_width = max(1, round(fraction * width))
    crop_height = max(1, round(fraction * height))
    left = round(min(max(0.0, point[0] - crop_width / 2), width - crop_width))
    top = round(min(max(0.0, point[1] - crop_height / 2), height - crop_height))
    return left, top, left + crop_width, top + crop_height


def _attention_crops(
    attention: torch.Tensor,
    grid: tuple[int, int],
    size: tuple[int, int],
    benchmark: str | None,
) -> list[tuple[int, int, int, int]]:
    width, height = size
    window = (1280, 720) if benchmark == "ui_vision" else (1288, 728)
    crop_width, crop_height = min(window[0], width), min(window[1], height)
    top = attention.topk(min(100, attention.numel())).indices.tolist()
    positions = [
        ((index % grid[1] + 0.5) / grid[1] * width,
         (index // grid[1] + 0.5) / grid[0] * height)
        for index in top
    ]
    ranked = []
    for x, y in positions:
        left = min(max(0.0, x - crop_width / 2), width - crop_width)
        upper = min(max(0.0, y - crop_height / 2), height - crop_height)
        box = (
            int(left),
            int(upper),
            int(left + crop_width),
            int(upper + crop_height),
        )
        coverage = sum(
            left <= px <= left + crop_width
            and upper <= py <= upper + crop_height
            for px, py in positions
        )
        ranked.append((coverage, box))
    ranked.sort(key=lambda row: row[0], reverse=True)
    selected = []
    for _, box in ranked:
        if box not in selected:
            selected.append(box)
        if len(selected) == 2:
            break
    return selected


def _distance(
    first: list[float], second: list[float], size: tuple[int, int]
) -> float:
    return math.hypot(
        (first[0] - second[0]) / size[0],
        (first[1] - second[1]) / size[1],
    )


def _area(box: tuple[int, int, int, int]) -> int:
    return (box[2] - box[0]) * (box[3] - box[1])
