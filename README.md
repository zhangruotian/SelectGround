# SelectGround

Official code and checkpoints for **Selection, Not Localization: Contrastive Training and Consensus Inference for GUI Grounding**.

SelectGround maps a screenshot and instruction to one click. It is trained with coordinate supervision and an element-aware target–distractor loss. ConGround is its training-free consensus inference method: it challenges the first full-view hypothesis, re-observes the two hypotheses at higher resolution, and resolves the observations with one fixed geometric rule.

## Results

| Model | Inference | ScreenSpot-Pro | UI-Vision | OSWorld-G |
|---|---|---:|---:|---:|
| SelectGround-8B | Direct | 64.96 | 38.68 | 70.00 |
| SelectGround-8B | + ConGround | **72.17** | **44.39** | **70.00** |
| SelectGround-30B-A3B | Direct | 65.91 | 38.69 | 72.35 |
| SelectGround-30B-A3B | + ConGround | **73.43** | **46.07** | **73.33** |

UI-Vision is the equal-weight mean of its basic, functional, and spatial element-grounding subsets. OSWorld-G uses the 510 target-bearing examples and excludes 54 refusal examples.

## Install

The training results below were reproduced with Python 3.12, CUDA 12.8, and the pinned packages in `requirements.txt`.

```bash
git clone https://github.com/zhangruotian/SelectGround.git
cd SelectGround
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
hf auth login
```

The repositories are private during artifact preparation. The same commands work without authentication after release.

## Ground one screenshot

SelectGround direct inference:

```bash
python infer.py \
  --model ruotian/SelectGround-8B \
  --image screenshot.png \
  --instruction "Click the Save button"
```

SelectGround + ConGround:

```bash
python infer.py \
  --model ruotian/SelectGround-8B \
  --image screenshot.png \
  --instruction "Click the Save button" \
  --conground
```

Use `ruotian/SelectGround-30B-A3B` for the 30B-A3B checkpoint. The returned `point` is in source-image pixels; `normalized_point` uses the `[0,1000]` coordinate system.

## Benchmarks

Download the official releases:

```bash
hf download lscpku/ScreenSpot-Pro --repo-type dataset --revision 211c2a6b5214b4dc8555a639a47575a9abd48c99 --local-dir data/screenspot-pro
hf download ServiceNow/ui-vision --repo-type dataset --revision 766c66aeffef16608d4916525902d9fb2598d7ce --local-dir data/ui-vision
git clone https://github.com/xlang-ai/OSWorld-G.git data/OSWorld-G
git -C data/OSWorld-G checkout daa6bd8e0e629f0917ad2984df930bf0bd967540
```

Run direct inference or append `--conground`:

```bash
python evaluate.py --model ruotian/SelectGround-8B --benchmark screenspot_pro --data data/screenspot-pro --output outputs/ssp.jsonl --conground
python evaluate.py --model ruotian/SelectGround-8B --benchmark ui_vision --data data/ui-vision --output outputs/uiv.jsonl --conground
python evaluate.py --model ruotian/SelectGround-8B --benchmark osworld_g --data data/OSWorld-G --output outputs/osw.jsonl --conground
```

The evaluator appends one result at a time and resumes from an existing output file.

Official benchmark pages: [ScreenSpot-Pro](https://huggingface.co/datasets/lscpku/ScreenSpot-Pro), [UI-Vision](https://huggingface.co/datasets/ServiceNow/ui-vision), and [OSWorld-G](https://github.com/xlang-ai/OSWorld-G/tree/main/benchmark).

## Train

The released [SelectGround-Data](https://huggingface.co/datasets/ruotian/SelectGround-Data) contains the exact screenshots and annotations used by both fixed recipes: 3,790 target–distractor pairs and 3,790 replay examples. The 8B recipe additionally uses the included 502-pair final refinement set with 502 matched replay examples.

The 8B model was trained on two L40S GPUs. Its two stages must run as separate processes: phase A writes the adapter together with the optimizer, scheduler, and per-rank RNG states; phase B restores those states before using the refinement set. Combining the stages in one process changes gradient accumulation at dataloader boundaries and does not reproduce the model.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True accelerate launch --mixed_precision bf16 --num_processes 2 train.py \
  --model 8b --stage phase_a --output outputs/SelectGround-8B-phase-a

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True accelerate launch --mixed_precision bf16 --num_processes 2 train.py \
  --model 8b --stage phase_b --checkpoint outputs/SelectGround-8B-phase-a --output outputs/SelectGround-8B
```

The 30B-A3B model was trained in one stage on four 80 GB A100 GPUs:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True accelerate launch --mixed_precision bf16 --num_processes 4 train.py --model 30b --output outputs/SelectGround-30B-A3B
```

Training is standard LoRA SFT. On mined examples, the loss additionally ranks the target region above the verified distractor and three hard UI regions. The training-only head aggregation is saved as `selection_head.pt`; inference uses only the trained grounding model.

With the pinned environment, two independent 8B clean runs reached 64.39 and 64.64 on ScreenSpot-Pro; one was also evaluated at 39.17 on UI-Vision and 69.02 on OSWorld-G. Independent 30B-A3B clean runs reached 64.20--64.83 on ScreenSpot-Pro, 38.57--38.90 on UI-Vision, and 70.98--71.76 on OSWorld-G. Small run-to-run differences remain across GPU nodes because the training kernels are not bitwise deterministic.

## Models and data

- [SelectGround-8B](https://huggingface.co/ruotian/SelectGround-8B)
- [SelectGround-30B-A3B](https://huggingface.co/ruotian/SelectGround-30B-A3B)
- [SelectGround-Data](https://huggingface.co/datasets/ruotian/SelectGround-Data)

The screenshots in the training repositories are the required subset of [Click-100K](https://huggingface.co/datasets/mlfoundations/Click-100k). Every example retains its upstream split and index.

## Citation

```bibtex
@article{selectground2026,
  title={Selection, Not Localization: Contrastive Training and Consensus Inference for GUI Grounding},
  author={Anonymous},
  year={2026}
}
```
