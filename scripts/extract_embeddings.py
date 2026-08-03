#!/usr/bin/env python3
"""Precompute the frozen QwenVL/SigLIP2 embeddings used by FrozenCal-K.

Input records are JSONL/JSON/CSV with group_id, source, instruction, edited.
The order is preserved exactly, and the cache index records every candidate.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frozencal.embeddings import extract_embeddings, save_cached_embeddings  # noqa: E402
from frozencal.io import read_records  # noqa: E402
from frozencal.image_inference import validate_image_records  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract frozen QwenVL/SigLIP2 embeddings for FrozenCal-K")
    parser.add_argument("--dataset", required=True, choices=["genai", "editreward_data", "editreward_bench"])
    parser.add_argument("--input", required=True, help="JSONL/JSON/CSV candidate manifest")
    parser.add_argument("--output-dir", required=True, help="Embedding cache directory")
    parser.add_argument("--qwen-model-path", required=True)
    parser.add_argument("--siglip-model-dir", required=True)
    parser.add_argument("--qwen-repo-root", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    rows = read_records(args.input)
    groups = validate_image_records(rows)
    started = time.perf_counter()
    tensors, timings = extract_embeddings(
        rows,
        qwen_model_path=args.qwen_model_path,
        siglip_model_dir=args.siglip_model_dir,
        qwen_repo_root=args.qwen_repo_root,
        batch_size=args.batch_size,
    )
    save_cached_embeddings(args.output_dir, tensors, rows)
    manifest = {
        "schema_version": 1,
        "dataset": args.dataset,
        "num_records": len(rows),
        "num_groups": len(groups),
        "qwen_model_path": str(args.qwen_model_path),
        "siglip_model_dir": str(args.siglip_model_dir),
        "batch_size": args.batch_size,
        "timings": {**timings, "total_seconds": time.perf_counter() - started},
    }
    output = Path(args.output_dir)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
