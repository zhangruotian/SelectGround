# SelectGround

Code and assets for **GUI Grounding as Selection under Observed and Latent Competition**.

SelectGround learns from target–distractor pairs with coordinate SFT and an auxiliary attention-based selection loss.
**Latent Competitor Revisit (LCR)** reuses the trained selector to choose two competitor crops, revisits the initial click, and selects one decoded coordinate using cross-view likelihood, selector evidence, and spatial agreement.

| SelectGround-8B | ScreenSpot-Pro | MMBench-GUI L2 | OSWorld-G |
|---|---:|---:|---:|
| Direct | 66.034 | 86.283 | 70.196 |
| + LCR | **73.182** | **88.008** | **71.961** |

The benchmarks contain 1,581 / 3,594 / 510 examples. OSWorld-G uses the target-bearing examples and original instructions.
[Model](https://huggingface.co/ruotian/SelectGround-8B) · [ClickContrast data](https://huggingface.co/datasets/ruotian/ClickContrast)

## Install

Use Python 3.12 and CUDA-capable NVIDIA GPUs. Dependencies are pinned to the experimental environment.

~~~bash
git clone --branch paper https://github.com/zhangruotian/SelectGround.git
cd SelectGround
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest test_core
~~~

Inference loads the released LoRA adapter and auxiliary selector on Qwen/Qwen3-VL-8B-Instruct, pinned at commit **0c351dd01ed87e9c1b53cbc748cba10e6187ff3b**.
The model and dataset downloads use their **paper** tags.
No parser or teacher is needed for training or inference from the released artifacts.

## Ground a screenshot

~~~bash
python infer.py --image screenshot.png --instruction "Click the Save button"
python infer.py --image screenshot.png --instruction "Click the Save button" --lcr
~~~

The output **point** is in original-image pixels; **normalized_point** uses 0–1000 coordinates.
Decoding is greedy with at most 32 tokens and an 8,847,360-pixel image budget.
For the closest match to the reference scores, use H200 GPUs with the pinned dependencies. GPU-dependent numerical differences can change greedy predictions.
For LCR, **--benchmark** chooses one of the three fixed comparison-weight presets; the default is **screenspot_pro**.

| Preset | Incumbent proximity | Selector evidence | Candidate agreement |
|---|---:|---:|---:|
| screenspot_pro | 0.850 | 0.275 | 0.300 |
| mmbench_gui_l2 | 0.475 | 1.625 | 0.300 |
| osworld_g | 0.125 | 0.150 | 0.300 |

These weights were tuned on the reported benchmarks.
LCR uses up to four visual grounding passes and additional candidate continuations with cached visual prefixes.
It returns one decoded point, not an average of points.

## Download benchmarks

~~~bash
hf download lscpku/ScreenSpot-Pro --repo-type dataset --revision 211c2a6b5214b4dc8555a639a47575a9abd48c99 --local-dir data/screenspot-pro
hf download OpenGVLab/MMBench-GUI --repo-type dataset --revision e27757e3910e0d5995b811f916b509b4e34a4690 --include L2_annotations.json MMBench-GUI-OfflineImages.zip --local-dir data/mmbench-gui
unzip data/mmbench-gui/MMBench-GUI-OfflineImages.zip -d data/mmbench-gui
git clone https://github.com/xlang-ai/OSWorld-G.git data/OSWorld-G
git -C data/OSWorld-G checkout daa6bd8e0e629f0917ad2984df930bf0bd967540
~~~

MMBench images must be under **data/mmbench-gui/offline_images/**.
ScreenSpot-Pro is read directly from its official parquet files.
Benchmark data remains subject to its respective source terms.

## Evaluate

~~~bash
python evaluate.py --benchmark screenspot_pro --data data/screenspot-pro --output outputs/ssp-direct.jsonl
python evaluate.py --benchmark mmbench_gui_l2 --data data/mmbench-gui --output outputs/mmb-direct.jsonl
python evaluate.py --benchmark osworld_g --data data/OSWorld-G --output outputs/osw-direct.jsonl
~~~

To evaluate LCR, append **--lcr** and use a new output path.
For example:

~~~bash
python evaluate.py --benchmark screenspot_pro --data data/screenspot-pro --lcr --output outputs/ssp-lcr.jsonl
~~~

**--model** accepts a downloaded or locally trained checkpoint directory.
**--limit 10** runs a smoke test.
Evaluation writes predictions incrementally and resumes when the same command is repeated.
Do not mix different checkpoints or inference settings in an output file.

The metrics file reports exact target accuracy and semantic misses.
A semantic miss is an incorrect on-screen point outside a centered 3x target-width/height neighborhood.
Both rates divide by all benchmark examples. Invalid outputs remain in the denominator.

For parallel evaluation, add **--num-shards N --shard I**, with a separate output file and GPU for each shard.
Then merge the complete shards:

~~~bash
python merge.py --inputs outputs/ssp-lcr-shard-*.jsonl --output outputs/ssp-lcr-full.jsonl
~~~

The two paper ablations use **--lcr --ablation no_competitors** or **--lcr --ablation no_selector**.
The first retains only the full-image and incumbent-revisit views.
The second uses fixed top-left/bottom-right competitor scopes and removes selector evidence.

## Reproduce training

Download all 8,584 ClickContrast rows and their screenshots:

~~~bash
hf download ruotian/ClickContrast --repo-type dataset --revision paper --local-dir data/clickcontrast
~~~

Run the fixed recipe on **two GPUs**, with one example per GPU microbatch and gradient accumulation 64.
The reference training hardware is two H200 GPUs. Keep the world size and accumulation unchanged.

~~~bash
CUDA_VISIBLE_DEVICES=0,1 bash reproduce.sh data/clickcontrast outputs/reproduction
~~~

The final checkpoint is **outputs/reproduction/final**. Evaluate it by passing that path to **--model**.

| Setting | Initial stage | Refinement |
|---|---|---|
| Data | All 4,292 pairs + 4,292 coverage rows | 502 refinement pairs + 502 refinement coverage rows |
| Coverage / pair microbatch cycle | 1 / 1 | 2 / 1 |
| Selected updates | 110 | 10 |
| Adapter learning rate | 3.45e-5 | 1e-6 |
| Cosine schedule horizon | 320 | 30 |
| Warmup updates | 10 | 10 |
| Auxiliary loss weight | 0.10 | 0.05 |

Both stages use seed 20260625, LoRA rank 64, alpha 128, dropout 0.05, and selector learning rate 1e-4.
The coordinate loss applies to every row.
Paired examples additionally use listwise target ranking over the distractor and up to three disjoint hard regions, plus a softplus pairwise margin of 0.3 with weight 0.5.
The selector learns weights over decoder layers 18–23 and all heads.
Only the adapters and selector are optimized; the visual encoder and merger remain frozen.

The reproduction script preserves process boundaries at initial-stage steps 100/110 and refinement steps 5/10.
Within a stage, it restores optimizer, scheduler, per-rank RNG, and data-stream position.
Refinement initializes from the initial-stage adapter and selector with a fresh optimizer and schedule.
The schedule horizons are not replaced by the selected update counts.

Checkpoints include training state so an interrupted segment can be restarted with **--resume** into a new output directory.
The released model files need only the adapter configuration, adapter weights, and selection head.
GPU kernels are not bitwise deterministic, so independent training can produce small numerical differences.
The paper checkpoint was selected using the three benchmarks; this script runs its fixed recipe without another hyperparameter search.

## Files

- **train.py**, **reproduce.sh**: the two-stage SFT + auxiliary-loss recipe.
- **selectground.py**: pinned model loading, prompt, and greedy coordinate decoding.
- **lcr.py**: competitor scopes and cross-view candidate selection.
- **infer.py**, **evaluate.py**, **merge.py**: screenshot inference and benchmark evaluation.
- **test_core.py**: coordinate, scoring, and LCR unit tests.

Code is Apache-2.0. Screenshot terms are described in the dataset card.
