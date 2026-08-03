# FrozenCal-K

FrozenCal-K is the K-adaptive calibration head described in the paper. QwenVL-Embedding-2B and SigLIP2 remain frozen; only a small linear preference scorer is calibrated. The scorer uses 12 absolute features and 12 within-group relative features. Separate heads are used for candidate groups with K=2, K=3, and K=4.

## Data Repository

Calibration and benchmark embedding files are hosted separately on ModelScope:

[Download FrozenCal-K data](https://modelscope.cn/datasets/naturezhanghn/FrozenCalK-data)

The dataset is organized as `editreward_data/qwen2b`, `editreward_data/siglip2`,
`editreward_bench/`, `genai_bench/`, and `calibration_cache/`. These files are not
committed to this code repository.

## Project map

```text
FrozenCalK/
|-- frozencal/                     # feature, embedding, and I/O library
|-- frozencal_k.py                 # calibration and variant training entry point
|-- infer_images.py                # image-record feature extraction and scoring
|-- embedding_infer.py             # cached GenAI/ERB benchmark inference
|-- prepare_editreward_data.py     # EditReward-Data shard/cache conversion
|-- scripts/
|   |-- extract_embeddings.py      # shared QwenVL + SigLIP2 pre-extractor
|   |-- extract_genai_embeddings.sh
|   |-- extract_editreward_data_embeddings.sh
|   `-- extract_editreward_bench_embeddings.sh
|   `-- train_frozencal.sh          # training shortcut
|-- run_embedding_inference.sh
|-- requirements.txt
`-- reproduction_data/             # local-only; ignored by Git
```

## Release contents

- `frozencal_k.py`: train a FrozenCal-K head from feature-level JSONL groups.
- `embedding_infer.py`: evaluate the released head on cached GenAI-Bench and EditReward-Bench embeddings.
- `prepare_editreward_data.py`: convert external EditReward-Data Qwen/SigLIP2 shards into feature-level JSONL for calibration.
- `run_embedding_inference.sh`: one-command benchmark inference.
- `infer_images.py`: extract features from image records and optionally score them with a compatible weight file.
- `frozencal/`: feature construction, embedding extraction, and I/O utilities.
- `scripts/extract_embeddings.py`: shared deterministic pre-extraction entry point for all three datasets.
- `scripts/extract_genai_embeddings.sh`, `scripts/extract_editreward_data_embeddings.sh`, `scripts/extract_editreward_bench_embeddings.sh`: dataset-specific shortcuts over the shared extractor.
- `reproduction_data/`: released QwenVL and SigLIP2 embedding caches and the released FrozenCal weights. Its embedding subdirectories explicitly name both the dataset and encoder, e.g. `editreward_bench_pair_qwen2b` and `editreward_bench_pair_siglip2`.
- `reproduced_results/embedding_inference/`: metrics written by the one-command inference.

## Calibration and reporting protocol

The paper uses two distinct calibration stages. First, EditReward-Data provides the base preference calibration data (12,000 constraints). Second, a target-domain protocol uses 30% of the target benchmark for lightweight adaptation. The remaining 70% is held out for reporting.

The 70% reporting split is not used for fitting, hyperparameter selection, standardization, early stopping, sign flipping, or model selection. The fixed split seeds are:

- GenAI-Bench: `20260728`
- EditReward-Bench K=2: `20260729`
- EditReward-Bench K=3: `20260730`
- EditReward-Bench K=4: `20260731`

The complete target benchmark embeddings are stored together in `reproduction_data/`; the 30%/70% split is applied by the evaluation code. The released weight file already contains the calibrated head, so `embedding_infer.py` is inference-only: it does not fit a new head and does not use the 30% subset to update the released weights.

## One-command cached inference

From any working directory, run:

```bash
bash /inspire/ssd/project/sais-bio/public/zhangziran/aaai_rice_project/FrozenCalK/run_embedding_inference.sh
```

The script reads the cached QwenVL/SigLIP2 embeddings and `frozencal_2b.json`, then writes `metrics.json` and `metrics.csv` under `reproduced_results/embedding_inference/`. The verified released-head results are:

| Metric | Accuracy |
| --- | ---: |
| GenAI-Bench | 71.21 |
| EditReward-Bench K=2 | 72.84 |
| EditReward-Bench K=3 | 33.33 |
| EditReward-Bench K=4 | 15.38 |
| EditReward-Bench overall | 40.52 |

These are held-out 70% reporting results. Accuracy on the 30% calibration portion is diagnostic only and must not be merged into the paper test result.

## Pre-extract embeddings

All pre-extraction scripts consume the same candidate manifest format used by `infer_images.py`:

```json
{"group_id":"g0", "source":"source.png", "instruction":"change the sky", "edited":"candidate.png", "candidate_id":"A", "model":"model_name"}
```

Every candidate in one group must share `group_id`, source image, and instruction. The extractor preserves input order, validates K=2/3/4 groups, computes the same QwenVL and SigLIP2 representations used by the scorer, and writes the standard cache files (`qwen_source.pt`, `qwen_text.pt`, `qwen_fused.pt`, `qwen_edited.pt`, `siglip_source.pt`, `siglip_text.pt`, `siglip_edited.pt`) plus an integrity `index.json` and timing `manifest.json`.

GenAI-Bench:

```bash
bash FrozenCalK/scripts/extract_genai_embeddings.sh \
  --input genai_records.jsonl \
  --output-dir caches/genai \
  --qwen-model-path models/Qwen3-VL-Embedding-2B \
  --siglip-model-dir models/siglip2-base-patch16-224 \
  --qwen-repo-root Qwen3-VL-Embedding
```

EditReward-Data:

```bash
bash FrozenCalK/scripts/extract_editreward_data_embeddings.sh \
  --input editreward_data_records.jsonl \
  --output-dir caches/editreward_data \
  --qwen-model-path models/Qwen3-VL-Embedding-2B \
  --siglip-model-dir models/siglip2-base-patch16-224 \
  --qwen-repo-root Qwen3-VL-Embedding
```

EditReward-Bench (K=2/3/4 groups):

```bash
bash FrozenCalK/scripts/extract_editreward_bench_embeddings.sh \
  --input editreward_bench_records.jsonl \
  --output-dir caches/editreward_bench \
  --qwen-model-path models/Qwen3-VL-Embedding-2B \
  --siglip-model-dir models/siglip2-base-patch16-224 \
  --qwen-repo-root Qwen3-VL-Embedding
```

The scripts do not download datasets or upload caches. For reproducibility, keep the manifest, model revisions, cache `index.json`, and `manifest.json` together. The embedding cache is an intermediate artifact; labels and calibration splits remain separate from feature extraction.

## Recalibration from feature JSONL

`frozencal_k.py` accepts one group per JSONL line. Each group must contain K candidates (`K` in {2,3,4}), a `split` field (`train`, `val`, or `test`), and either preference/tie edges, human scores, or ranks. The training input is feature-level data; raw benchmark embedding shards are not accepted directly by this entry point.

The `--variant` interface exposes all three paper scorer variants:

- `--variant k` (default): 24-dimensional input with separate `w2`, `w3`, and `w4` heads (`FrozenCal-K`).
- `--variant abs12-single`: 12 absolute features with one shared head (`FrozenCal-Abs12-Single`).
- `--variant rel24-shared`: 24 absolute/relative features with one shared head (`FrozenCal-Rel24-Shared`).

For example:

```bash
python FrozenCalK/frozencal_k.py \
  --input groups.jsonl \
  --output abs12_weights.json \
  --variant abs12-single
```

The image inference entry point reads the `variant` field in these weight files and applies the matching 12- or 24-dimensional head.

## Training workflow

The complete local workflow has four explicit stages. The encoder weights stay frozen in every stage.

### 1. Prepare candidate manifests

Create one JSONL record per edited candidate. All candidates in a group must share `group_id`, `source`, and `instruction`:

```json
{"group_id":"case-000", "source":"source.png", "instruction":"change the sky", "edited":"edited_A.png", "candidate_id":"A", "model":"model_A"}
```

Use separate manifests for GenAI-Bench, EditReward-Data, and EditReward-Bench. Keep the manifests and model revisions with the experiment; they determine the cache order.

### 2. Extract frozen embeddings

Run the dataset-specific shortcut. Each shortcut calls the same extractor and writes QwenVL/SigLIP2 tensors plus an integrity index:

```bash
bash FrozenCalK/scripts/extract_editreward_data_embeddings.sh \
  --input editreward_data_records.jsonl \
  --output-dir caches/editreward_data \
  --qwen-model-path models/Qwen3-VL-Embedding-2B \
  --siglip-model-dir models/siglip2-base-patch16-224 \
  --qwen-repo-root Qwen3-VL-Embedding
```

The same command pattern applies to `extract_genai_embeddings.sh` and `extract_editreward_bench_embeddings.sh`. This stage does not use human labels and does not train a scorer.

### 3. Build calibration JSONL

For EditReward-Data, convert the prepared 12,000-record cache directly when available:

```bash
python FrozenCalK/prepare_editreward_data.py \
  --cache results/mainpaper_frozencal_calibration/calibration_raw12_n12000_seed42.pt \
  --output editreward_data_calibration.jsonl
```

Alternatively, read QwenVL/SigLIP2 EditReward-Data shards with `--qwen-dir` and `--siglip-dir`. The converter creates 9,510 directional and 2,490 tie constraints and a deterministic train/validation split.

### 4. Train a scorer

Train the paper's three variants with the same entry point:

```bash
# FrozenCal-K: separate K=2, K=3, K=4 heads
bash FrozenCalK/scripts/train_frozencal.sh \
  --input editreward_data_calibration.jsonl \
  --output weights_k.json \
  --variant k \
  --k-values 2

# Absolute-only shared head
bash FrozenCalK/scripts/train_frozencal.sh \
  --input groups.jsonl \
  --output weights_abs12.json \
  --variant abs12-single

# Absolute plus relative features with one shared head
bash FrozenCalK/scripts/train_frozencal.sh \
  --input groups.jsonl \
  --output weights_rel24.json \
  --variant rel24-shared
```

The trainer standardizes absolute features using training candidates only, computes group-relative z-scores, builds directional and tie constraints, searches the listed hyperparameters on validation groups, and reports untouched test groups when present. `--enforce-targets` applies the validation-only gates (>65%, >30%, >10% for K=2/3/4); it never changes the test metric.

### 5. Target-domain adaptation and reporting

For the paper protocol, first fit the base scorer on EditReward-Data, then use only 30% of each target benchmark for lightweight target calibration. GenAI and EditReward-Bench calibration portions are jointly used for target search. The remaining 70% is held out and evaluated once after model selection. The released cached inference command reports this final 70% split:

```bash
bash FrozenCalK/run_embedding_inference.sh
```

Do not merge calibration and reporting portions into one test score. The target-search implementation used to produce the released paper weights is retained in the research workspace; the public release provides the deterministic feature extraction, base calibration, variant training, and released-weight inference paths without shipping benchmark data.

Example:

```bash
python FrozenCalK/frozencal_k.py \
  --input groups.jsonl \
  --output weights.json \
  --report calibration_report.json
```

## EditReward-Data compatibility

The training entry point accepts feature-level JSONL rather than raw `.pt` shards. Use the converter when the external EditReward-Data caches are available:

```bash
python FrozenCalK/prepare_editreward_data.py \
  --qwen-dir /path/to/data/editreward_data_embeddings/2b \
  --siglip-dir /path/to/data/siglip2_features/editreward_data \
  --output editreward_data_calibration.jsonl

python FrozenCalK/frozencal_k.py \
  --input editreward_data_calibration.jsonl \
  --output editreward_data_k2_weights.json \
  --k-values 2
```

By default the converter selects 9,510 directional and 2,490 tie records from the first 68 matching shards, using seed `42`, matching the paper's 12,000-record calibration scale. The converter verifies Qwen/SigLIP2 metadata alignment and computes the same 12 absolute features used by the scorer. `--k-values 2` is intended for the K=2 base-calibration stage; K=3/K=4 target heads require corresponding multi-candidate groups.

This path is protocol-compatible with the paper, but exact published weights additionally depend on the paper's fixed calibration split, initialization, search settings, and target-domain 30% adaptation. The external 6.7 GB EditReward-Data cache is intentionally not included in this repository.

If the prepared 12,000-record cache from the paper is available, it can be used without rereading the embedding shards:

```bash
python FrozenCalK/prepare_editreward_data.py \
  --cache results/mainpaper_frozencal_calibration/calibration_raw12_n12000_seed42.pt \
  --output editreward_data_calibration.jsonl

python FrozenCalK/frozencal_k.py \
  --input editreward_data_calibration.jsonl \
  --output editreward_data_k2_weights.json \
  --k-values 2
```

The existing cache produced 12,000 groups and the current trainer reached 68.92% K=2 validation accuracy (827/1,200). This is a base-calibration diagnostic, not the paper's final target-domain test score.

With `--enforce-targets`, calibration fails unless validation exceeds the release criteria: strictly greater than 65% for K=2, 30% for K=3, and 10% for K=4. These are validation search gates, not test-set claims.

## Scope and reproducibility note

The cached inference entry point reproduces the released FrozenCal-K benchmark row. It does not retrain the QwenVL or SigLIP2 encoders, and it does not reimplement external baseline models or every ablation row from the paper. Full target recalibration requires the EditReward-Data feature/label files and a newly specified calibration split.
