#!/usr/bin/env python
"""Extract FrozenCal features from images and optionally score candidates."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from frozencal.embeddings import (
    align_group_embeddings,
    extract_embeddings,
    load_cached_embeddings,
    save_cached_embeddings,
)
from frozencal.features import ABS_FEATURE_NAMES, candidate_features
from frozencal.io import read_records, write_rows
from frozencal_k import FEATURE_DIM, SUPPORTED_K, frozen_features, grouped_indices


CACHE_TENSORS = (
    "qwen_source",
    "qwen_text",
    "qwen_fused",
    "qwen_edited",
    "siglip_source",
    "siglip_text",
    "siglip_edited",
)


def validate_image_records(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    groups = grouped_indices(rows)
    for index, row in enumerate(rows):
        for key in ("source", "edited"):
            if not Path(row[key]).is_file():
                raise ValueError(f"Record {index}: {key} image does not exist: {row[key]}")
    for group_id, indices in groups.items():
        k = len(indices)
        if k not in SUPPORTED_K:
            raise ValueError(f"Group {group_id!r}: K={k} is unsupported; expected 2, 3, or 4")
        sources = {str(rows[index]["source"]) for index in indices}
        instructions = {str(rows[index]["instruction"]) for index in indices}
        if len(sources) != 1 or len(instructions) != 1:
            raise ValueError(f"Group {group_id!r}: source and instruction must be identical within the group")
    return groups


def expected_cache_index(rows: list[dict[str, Any]], groups: dict[str, list[int]]) -> dict[str, Any]:
    first_rows = [rows[indices[0]] for indices in groups.values()]
    return {
        "num_records": len(rows),
        "num_groups": len(groups),
        "groups": [
            {
                "group_id": row["group_id"],
                "source": row["source"],
                "instruction": row["instruction"],
            }
            for row in first_rows
        ],
        "records": [
            {
                "group_id": row["group_id"],
                "edited": row["edited"],
                "candidate_id": row.get("candidate_id", ""),
                "model": row.get("model", ""),
            }
            for row in rows
        ],
    }


def load_or_extract(args: argparse.Namespace, rows: list[dict[str, Any]], groups: dict[str, list[int]]):
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        tensor_paths = [cache_dir / f"{name}.pt" for name in CACHE_TENSORS]
        index_path = cache_dir / "index.json"
        present = [path.is_file() for path in [*tensor_paths, index_path]]
        if all(present):
            cached_index = json.loads(index_path.read_text(encoding="utf-8"))
            if cached_index != expected_cache_index(rows, groups):
                raise ValueError("Embedding cache index does not match the input records")
            started = time.perf_counter()
            return load_cached_embeddings(cache_dir), {"cache_load_seconds": time.perf_counter() - started}
        if any(present):
            raise ValueError(f"Embedding cache is incomplete: {cache_dir}")

    if not args.qwen_model_path or not args.siglip_model_dir:
        raise ValueError("Model paths are required when a complete matching cache is unavailable")
    tensors, timings = extract_embeddings(
        rows,
        qwen_model_path=args.qwen_model_path,
        siglip_model_dir=args.siglip_model_dir,
        qwen_repo_root=args.qwen_repo_root,
        batch_size=args.batch_size,
    )
    if args.cache_dir:
        save_cached_embeddings(args.cache_dir, tensors, rows)
    return tensors, timings


def score_rows(
    rows: list[dict[str, Any]],
    groups: dict[str, list[int]],
    absolute: torch.Tensor,
    weights_path: str | Path,
) -> list[dict[str, Any]]:
    payload = json.loads(Path(weights_path).read_text(encoding="utf-8"))
    if int(payload.get("feature_dim_absolute", -1)) != FEATURE_DIM:
        raise ValueError("Weight file is incompatible with the 12-dimensional image features")
    variant = payload.get("variant", "k")
    if variant not in {"k", "abs12-single", "rel24-shared"}:
        raise ValueError(f"Unknown FrozenCal variant: {variant}")
    mean = torch.tensor(payload["feature_mean"], dtype=torch.float32)
    std = torch.tensor(payload["feature_std"], dtype=torch.float32)
    feature_rows = [{**row, "features": absolute[index].tolist()} for index, row in enumerate(rows)]
    matrix = frozen_features(feature_rows, groups, mean, std)
    if variant == "abs12-single":
        matrix = matrix[:, :FEATURE_DIM]
    output = [dict(row) for row in rows]
    for indices in groups.values():
        k = len(indices)
        key = "shared" if variant in {"abs12-single", "rel24-shared"} else f"w{k}"
        weight_values = payload["weights"].get(key)
        expected_dim = FEATURE_DIM if variant == "abs12-single" else FEATURE_DIM * 2
        if weight_values is None or len(weight_values) != expected_dim:
            raise ValueError(f"Weight file does not contain a valid {key} head")
        weight = torch.tensor(weight_values, dtype=torch.float32)
        index = torch.tensor(indices, dtype=torch.long)
        scores = matrix[index] @ weight
        order = torch.argsort(scores, descending=True).tolist()
        ranks = [0] * k
        for rank, local_index in enumerate(order, start=1):
            ranks[local_index] = rank
        for local_index, row_index in enumerate(indices):
            output[row_index]["frozencal_score"] = float(scores[local_index])
            output[row_index]["rank"] = ranks[local_index]
            output[row_index]["head_k"] = k
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="FrozenCal-K image-to-score inference")
    parser.add_argument("--input", required=True, help="JSONL/JSON/CSV image candidate records")
    parser.add_argument("--weights", default=None, help="Weights produced by frozencal_k.py train")
    parser.add_argument("--output", default=None, help="Scored JSONL/JSON/CSV output")
    parser.add_argument("--feature-output", default=None, help="Optional 12-dimensional feature JSONL/JSON/CSV")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--qwen-model-path", default=None)
    parser.add_argument("--qwen-repo-root", default=None)
    parser.add_argument("--siglip-model-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timing-output", default=None)
    args = parser.parse_args()

    if bool(args.weights) != bool(args.output):
        parser.error("--weights and --output must be provided together")
    if not args.output and not args.feature_output:
        parser.error("provide --feature-output, or provide both --weights and --output")

    try:
        rows = read_records(args.input)
        groups = validate_image_records(rows)
        tensors, timings = load_or_extract(args, rows, groups)
        aligned = align_group_embeddings(rows, tensors)
        absolute = candidate_features(
            aligned["qwen_source"],
            aligned["qwen_text"],
            aligned["qwen_fused"],
            aligned["qwen_edited"],
            aligned["siglip_source"],
            aligned["siglip_text"],
            aligned["siglip_edited"],
        )
        if absolute.shape != (len(rows), FEATURE_DIM):
            raise ValueError(f"Unexpected absolute feature shape: {tuple(absolute.shape)}")

        if args.feature_output:
            feature_rows = []
            for index, row in enumerate(rows):
                item = dict(row)
                item["features"] = [float(value) for value in absolute[index]]
                for feature_index, name in enumerate(ABS_FEATURE_NAMES):
                    item[name] = float(absolute[index, feature_index])
                feature_rows.append(item)
            write_rows(args.feature_output, feature_rows)
            print(f"Features written to {args.feature_output}")

        if args.output:
            write_rows(args.output, score_rows(rows, groups, absolute, args.weights))
            print(f"Scores written to {args.output}")

        if args.timing_output:
            Path(args.timing_output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.timing_output).write_text(json.dumps(timings, indent=2) + "\n", encoding="utf-8")
        return 0
    except (ValueError, KeyError, OSError, ImportError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
