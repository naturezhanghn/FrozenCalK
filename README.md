# FrozenCal-K

FrozenCal-K is the K-adaptive calibration head described in the paper. QwenVL-Embedding-2B and SigLIP2 remain frozen; only a small linear preference scorer is calibrated. The scorer uses 12 absolute features and 12 within-group relative features. Separate heads are used for candidate groups with K=2, K=3, and K=4.

## Release contents

- `frozencal_k.py`: train a FrozenCal-K head from feature-level JSONL groups.
- `embedding_infer.py`: evaluate the released head on cached GenAI-Bench and EditReward-Bench embeddings.
- `run_embedding_inference.sh`: one-command benchmark inference.
- `infer_images.py`: extract features from image records and optionally score them with a compatible weight file.
- `frozencal/`: feature construction, embedding extraction, and I/O utilities.
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

## Recalibration from feature JSONL

`frozencal_k.py` accepts one group per JSONL line. Each group must contain K candidates (`K` in {2,3,4}), a `split` field (`train`, `val`, or `test`), and either preference/tie edges, human scores, or ranks. The training input is feature-level data; raw benchmark embedding shards are not accepted directly by this entry point.

Example:

```bash
python FrozenCalK/frozencal_k.py \
  --input groups.jsonl \
  --output weights.json \
  --report calibration_report.json
```

With `--enforce-targets`, calibration fails unless validation exceeds the release criteria: strictly greater than 65% for K=2, 30% for K=3, and 10% for K=4. These are validation search gates, not test-set claims.

## Scope and reproducibility note

The cached inference entry point reproduces the released FrozenCal-K benchmark row. It does not retrain the QwenVL or SigLIP2 encoders, and it does not reimplement external baseline models or every ablation row from the paper. Full target recalibration requires the EditReward-Data feature/label files and a newly specified calibration split.
