# SelectGround

Code and assets for **GUI Grounding as Selection under Observed and Latent Competition**.
SelectGround is trained on target–distractor pairs with coordinate SFT and an auxiliary selection loss. Latent Competitor Revisit (LCR) revisits likely competitor regions at inference time.

| Released SelectGround-8B | ScreenSpot-Pro | MMBench-GUI L2 | OSWorld-G |
|---|---:|---:|---:|
| Direct | 66.034 | 86.283 | 70.196 |
| + LCR | **73.182** | **88.008** | **71.961** |

[🤗 Model](https://huggingface.co/ruotian/SelectGround-8B) · [🤗 Data](https://huggingface.co/datasets/ruotian/ClickContrast)

## Install

~~~bash
git clone https://github.com/zhangruotian/SelectGround.git
cd SelectGround
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

## Download the model

~~~bash
hf download ruotian/SelectGround-8B --revision paper --local-dir models/SelectGround-8B
MODEL=models/SelectGround-8B
~~~

## Download benchmarks

~~~bash
hf download lscpku/ScreenSpot-Pro --repo-type dataset --revision 211c2a6b5214b4dc8555a639a47575a9abd48c99 --local-dir data/screenspot-pro

hf download OpenGVLab/MMBench-GUI --repo-type dataset --revision e27757e3910e0d5995b811f916b509b4e34a4690 --include L2_annotations.json MMBench-GUI-OfflineImages.zip --local-dir data/mmbench-gui
unzip data/mmbench-gui/MMBench-GUI-OfflineImages.zip -d data/mmbench-gui

git clone https://github.com/xlang-ai/OSWorld-G.git data/OSWorld-G
git -C data/OSWorld-G checkout daa6bd8e0e629f0917ad2984df930bf0bd967540
~~~

## Evaluate Direct and LCR

Run all three Direct evaluations:

~~~bash
python evaluate.py --model "$MODEL" --benchmark screenspot_pro --data data/screenspot-pro --output outputs/ssp-direct.jsonl
python evaluate.py --model "$MODEL" --benchmark mmbench_gui_l2 --data data/mmbench-gui --output outputs/mmb-direct.jsonl
python evaluate.py --model "$MODEL" --benchmark osworld_g --data data/OSWorld-G --output outputs/osw-direct.jsonl
~~~

Run all three LCR evaluations:

~~~bash
python evaluate.py --model "$MODEL" --benchmark screenspot_pro --data data/screenspot-pro --lcr --output outputs/ssp-lcr.jsonl
python evaluate.py --model "$MODEL" --benchmark mmbench_gui_l2 --data data/mmbench-gui --lcr --output outputs/mmb-lcr.jsonl
python evaluate.py --model "$MODEL" --benchmark osworld_g --data data/OSWorld-G --lcr --output outputs/osw-lcr.jsonl
~~~

## Single-image inference

~~~bash
python infer.py --model "$MODEL" --image screenshot.png --instruction "Click the Save button"
python infer.py --model "$MODEL" --image screenshot.png --instruction "Click the Save button" --lcr
~~~

## Reproduce training

The released paper checkpoint predates our fully frozen environment, and its complete low-level execution state—including the exact data order, software and CUDA kernel versions, and distributed operation order—was not preserved, so its exact optimization trajectory cannot be replayed from scratch. We therefore release it for exact paper evaluation and provide a pinned deterministic recipe below that reproduces a new checkpoint byte-for-byte.

Reference training uses **2 NVIDIA H100 GPUs**. Reference evaluation uses **1 NVIDIA H200 GPU**.

~~~bash
hf download ruotian/ClickContrast --repo-type dataset --revision paper --local-dir data/clickcontrast
CUDA_VISIBLE_DEVICES=0,1 bash reproduce.sh data/clickcontrast outputs/reproduction
~~~

The reproduced checkpoint is written to **outputs/reproduction**. Its reference H200 results are `65.528 / 85.671 / 69.804` with Direct inference and `71.917 / 87.730 / 71.961` with LCR on ScreenSpot-Pro, MMBench-GUI L2, and OSWorld-G, respectively.

Code is Apache-2.0. Screenshot terms are described in the dataset card.
