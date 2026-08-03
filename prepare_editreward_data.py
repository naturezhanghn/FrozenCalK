#!/usr/bin/env python3
"""Convert EditReward-Data Qwen/SigLIP2 shards to FrozenCal feature JSONL.

The converter keeps the source shards outside the release repository and
materializes only the selected feature-level calibration records.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from frozencal.features import candidate_features


def shard_ids(q_dir: Path, s_dir: Path) -> list[int]:
    q = {int(p.stem.split("_")[-1]) for p in q_dir.glob("shard_*.pt")}
    s = {int(p.stem.split("_")[-1]) for p in s_dir.glob("shard_*.pt")}
    if not q or not s:
        raise FileNotFoundError("No matching Qwen/SigLIP2 EditReward-Data shards found")
    return sorted(q & s)


def select_rows(q_dir: Path, ids: list[int], directional: int, ties: int, seed: int) -> list[tuple[int, int, bool]]:
    direction_rows: list[tuple[int, int]] = []
    tie_rows: list[tuple[int, int]] = []
    for shard_id in ids:
        payload = torch.load(q_dir / f"shard_{shard_id:04d}.pt", map_location="cpu", weights_only=False)
        for row, meta in enumerate(payload["metadata"]):
            if meta.get("vote_type") in {"leftvote", "rightvote"}:
                direction_rows.append((shard_id, row))
            elif meta.get("vote_type") == "tie":
                tie_rows.append((shard_id, row))
    rng = random.Random(seed)
    rng.shuffle(direction_rows); rng.shuffle(tie_rows)
    if len(direction_rows) < directional or len(tie_rows) < ties:
        raise ValueError(f"Requested {directional} directional and {ties} tie rows, but only {len(direction_rows)} and {len(tie_rows)} are available")
    selected = [(s, r, False) for s, r in direction_rows[:directional]] + [(s, r, True) for s, r in tie_rows[:ties]]
    rng.shuffle(selected)
    return selected


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Prepare EditReward-Data features for frozencal_k.py")
    parser.add_argument("--qwen-dir", required=True, help="Directory containing Qwen shard_XXXX.pt files")
    parser.add_argument("--siglip-dir", required=True, help="Directory containing SigLIP2 shard_XXXX.pt files")
    parser.add_argument("--output", required=True, help="Output feature JSONL")
    parser.add_argument("--directional", type=int, default=9510)
    parser.add_argument("--ties", type=int, default=2490)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    args = parser.parse_args()
    q_dir, s_dir = Path(args.qwen_dir), Path(args.siglip_dir)
    selected = select_rows(q_dir, shard_ids(q_dir, s_dir)[:68], args.directional, args.ties, args.seed)
    by_shard: dict[int, list[tuple[int, bool]]] = defaultdict(list)
    for shard, row, is_tie in selected:
        by_shard[shard].append((row, is_tie))

    records = []
    for shard in sorted(by_shard):
        q = torch.load(q_dir / f"shard_{shard:04d}.pt", map_location="cpu", weights_only=False)
        s = torch.load(s_dir / f"shard_{shard:04d}.pt", map_location="cpu", weights_only=False)
        q_keys = [m["unique_key"] for m in q["metadata"]]
        s_keys = [m["unique_key"] for m in s["metadata"]]
        if q_keys != s_keys:
            raise ValueError(f"Shard {shard}: Qwen and SigLIP2 metadata order differs")
        rows = by_shard[shard]
        index = torch.tensor([row for row, _ in rows], dtype=torch.long)
        left = candidate_features(q["source"][index], q["text"][index], q["fused"][index], q["left"][index], s["source"][index], s["text"][index], s["left"][index])
        right = candidate_features(q["source"][index], q["text"][index], q["fused"][index], q["right"][index], s["source"][index], s["text"][index], s["right"][index])
        for local, (row, is_tie) in enumerate(rows):
            meta = q["metadata"][row]
            item = {
                "group_id": str(meta["unique_key"]),
                "split": "val" if (local % max(1, round(1 / args.validation_fraction))) == 0 else "train",
                "candidates": [
                    {"candidate_id": "left", "features": [float(x) for x in left[local]], "human_score": float(meta["left_overall_score"])},
                    {"candidate_id": "right", "features": [float(x) for x in right[local]], "human_score": float(meta["right_overall_score"])},
                ],
                "metadata": {"vote_type": meta.get("vote_type"), "source_key": meta.get("key"), "is_tie": bool(is_tie)},
            }
            records.append(item)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item) + "\n")
    print(json.dumps({"output": str(output), "groups": len(records), "directional": args.directional, "ties": args.ties}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
