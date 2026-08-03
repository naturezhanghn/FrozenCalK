# FrozenCal-K

Lightweight preference calibration for image-edit evaluation. QwenVL-Embedding-2B and SigLIP2 stay frozen; only a K-specific linear head is fitted.

**Project homepage:** [FrozenCal-K on GitHub](https://github.com/naturezhanghn/FrozenCalK)  
**Online demo:** [ModelScope Studio](https://modelscope.cn/studios/naturezhanghn/FrozenCalK-demo)  
**Paper page:** [`docs/index.html`](https://github.com/naturezhanghn/FrozenCalK/blob/main/docs/index.html)  
**Data and release weights:** [ModelScope dataset](https://modelscope.cn/datasets/naturezhanghn/FrozenCalK-data)

## Layout

```text
frozencal/                 core features, training, and inference modules
scripts/                   embedding and training shortcuts
demo/                      Gradio app and Jupyter playground
docs/index.html            paper companion page
```

## Install

```bash
pip install -r requirements.txt
```

Download the release head and benchmark caches from ModelScope. Keep them under
`reproduction_data/` (this directory is intentionally excluded from GitHub).

```bash
modelscope download --dataset naturezhanghn/FrozenCalK-data \
  weights/frozencal_2b.json --local_dir reproduction_data
```

## Reproduce cached metrics

```bash
bash run_embedding_inference.sh \
  --data-root reproduction_data \
  --weights reproduction_data/weights/frozencal_2b.json
```

The released head reports **71.21** GenAI-Bench, **72.84 / 33.33 / 15.38** on
EditReward-Bench K=2/3/4, and **40.52** overall on the held-out reporting split.

## Train or extract

```bash
bash scripts/train_frozencal.sh --input groups.jsonl --output weights.json --variant k
bash scripts/extract_genai_embeddings.sh --input records.jsonl --output-dir cache \
  --qwen-model-path models/Qwen3-VL-Embedding-2B \
  --siglip-model-dir models/siglip2-base-patch16-224
```

The trainer supports `k`, `abs12-single`, and `rel24-shared`. The encoder
weights remain frozen; only the small calibration head is optimized.

## Demo locally

```bash
cd demo
pip install -r requirements.txt
python app.py
```

The demo ranks K candidates from their 12 frozen absolute features. Full
encoder-side image inference is available in `frozencal/image_inference.py`.
