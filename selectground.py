"""SelectGround-8B: pinned backbone, coordinate decoding, and LCR."""
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv

BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
BASE_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
PROMPT = """You are an expert GUI grounding model.
Given a screenshot and an instruction, point to the UI element that should be clicked.
Return only one point as [x, y], where x and y are normalized integers from 0 to 1000 relative to the full image.
For an element with area, return the center point.
Instruction: {instruction}"""


@dataclass
class Prefix:
    cache: DynamicCache
    logits: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    length: int


def prediction(raw, size):
    match = re.search(
        r"[\[(（]\s*(?:x\s*=\s*)?(-?\d+(?:\.\d+)?)\s*[,，]\s*"
        r"(?:y\s*=\s*)?(-?\d+(?:\.\d+)?)\s*[\])）]", raw, re.IGNORECASE,
    )
    normalized = [float(match[1]), float(match[2])] if match else None
    point = [normalized[0] / 1000 * size[0], normalized[1] / 1000 * size[1]] if normalized else None
    return {"point": point, "normalized_point": normalized, "raw_response": raw}


class SelectGround:
    def __init__(self, checkpoint="ruotian/SelectGround-8B"):
        path = Path(checkpoint)
        if not path.is_dir():
            path = Path(snapshot_download(checkpoint, revision="paper"))
        base = AutoModelForImageTextToText.from_pretrained(
            BASE_MODEL, revision=BASE_REVISION, dtype=torch.bfloat16,
            device_map="auto", attn_implementation="sdpa",
        )
        self.model = PeftModel.from_pretrained(base, path).eval()
        self.model.config.use_cache = True
        self.processor = AutoProcessor.from_pretrained(
            BASE_MODEL, revision=BASE_REVISION, min_pixels=3136, max_pixels=8847360,
        )
        self.device = next(self.model.parameters()).device
        self.core = self.model.get_base_model().model
        head = torch.load(path / "selection_head.pt", map_location="cpu", weights_only=True)
        self.layers = head["layers"]
        self.weights = head["layer_head_weights"].float().flatten().softmax(0)

    def predict(self, image, instruction, *, lcr=False, benchmark="screenspot_pro", ablation="full"):
        source = image.convert("RGB") if isinstance(image, Image.Image) else Image.open(image).convert("RGB")
        if lcr:
            from lcr import revisit
            return revisit(self, source, instruction, benchmark, ablation)
        result, _, _, _ = self.observe(source, instruction)
        return {"method": "SelectGround", **result}

    def inputs(self, image, instruction):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT.format(instruction=instruction)},
            ],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        return self.processor(
            text=[text], images=images, videos=videos, padding=True, return_tensors="pt",
        ).to(self.device)

    @torch.inference_mode()
    def observe(self, image, instruction, capture=False):
        inputs = self.inputs(image, instruction)
        ids = inputs["input_ids"]
        length = ids.shape[1]
        merge = self.core.config.vision_config.spatial_merge_size
        grid = tuple(int(v) // merge for v in inputs["image_grid_thw"][0, 1:])
        positions, _ = self.core.get_rope_index(
            ids, inputs.get("image_grid_thw"), inputs.get("video_grid_thw"),
            attention_mask=inputs["attention_mask"],
        )
        cache = DynamicCache(config=self.core.language_model.config)
        values, handles = {}, []
        if capture:
            visual = torch.nonzero(ids[0] == self.core.config.image_token_id, as_tuple=False).flatten()

            def hook(layer):
                def collect(module, args, kwargs, output):
                    hidden = kwargs["hidden_states"]
                    shape = (*hidden.shape[:-1], -1, module.head_dim)
                    query = module.q_norm(module.q_proj(hidden).view(shape)).transpose(1, 2)
                    keys = module.k_norm(module.k_proj(hidden).view(shape)).transpose(1, 2)
                    query, keys = apply_rotary_pos_emb(query, keys, *kwargs["position_embeddings"])
                    keys = repeat_kv(keys, module.num_key_value_groups).index_select(2, visual.to(keys.device))
                    values[layer] = (query[:, :, -1:, :] * keys).sum(-1)[0].float() * module.scaling
                return collect

            handles = [
                self.core.language_model.layers[layer].self_attn.register_forward_hook(hook(layer), with_kwargs=True)
                for layer in self.layers
            ]
        try:
            output = self.model(
                **inputs, past_key_values=cache, position_ids=positions,
                cache_position=torch.arange(length, device=self.device), use_cache=True, logits_to_keep=1,
            )
        finally:
            for handle in handles:
                handle.remove()
        scores = None
        if capture:
            stacked = torch.stack([values[layer] for layer in self.layers])
            weights = self.weights.to(stacked.device).view(stacked.shape[:2])
            scores = (stacked * weights[:, :, None]).sum((0, 1)).cpu()
        logits = output.logits[:, -1, :].detach()
        raw = self.decode(logits, cache, positions[:, :, -1:] + 1)
        cache.crop(length)
        return prediction(raw, image.size), Prefix(cache, logits, positions, inputs["attention_mask"], length), scores, grid

    def decode(self, logits, cache, positions):
        tokenizer = self.processor.tokenizer
        stop = {tokenizer.eos_token_id, tokenizer.pad_token_id}
        generated = []
        for _ in range(32):
            token = int(logits.argmax())
            if token in stop:
                break
            generated.append(token)
            output = self.model(
                input_ids=torch.tensor([[token]], device=self.device), past_key_values=cache,
                position_ids=positions, cache_position=torch.tensor([cache.get_seq_length()], device=self.device),
                use_cache=True, logits_to_keep=1,
            )
            cache = output.past_key_values
            logits = output.logits[:, -1, :]
            positions = positions + 1
        return self.processor.decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        ).strip()
