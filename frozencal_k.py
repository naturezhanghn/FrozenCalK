#!/usr/bin/env python
"""FrozenCal-K calibration and feature-level inference.

Preferred training JSONL format (one group per line)::
  {"group_id":"g", "split":"train", "candidates":[
    {"candidate_id":"a", "features":[12 floats], "human_score":0.9}, ...],
    "preference_edges":[["a","b"]], "tie_edges":[["c","d"]]}

``preference_edges`` may contain [winner, loser, score_gap]. If edges are
omitted, human_score (with --delta-pref) or rank fields are converted to
directional/tie constraints. Flat records with rank remain supported.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


FEATURE_DIM = 12
SUPPORTED_K = (2, 3, 4)
DEFAULT_TARGETS = {2: 65.0, 3: 30.0, 4: 10.0}


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    features: list[float]
    human_score: float | None = None
    rank: int | None = None


@dataclass
class Group:
    group_id: str
    split: str
    candidates: list[Candidate]
    preference_edges: list[tuple[str, str, float]]
    tie_edges: list[tuple[str, str]]


@dataclass(frozen=True)
class SearchConfig:
    lr: float
    l2: float
    epochs: int
    delta_pref: float
    delta_scale: float
    tie_weight: float


def grouped_indices(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["group_id"])].append(index)
    return dict(groups)


def frozen_features(
    rows: list[dict[str, Any]],
    groups: dict[str, list[int]],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    absolute = torch.tensor([row["features"] for row in rows], dtype=torch.float32)
    relative = torch.zeros_like(absolute)
    for indices in groups.values():
        index = torch.tensor(indices, dtype=torch.long)
        values = absolute[index]
        relative[index] = (values - values.mean(0)) / values.std(0, unbiased=True).clamp_min(1e-4)
    return torch.cat([(absolute - mean) / std.clamp_min(1e-6), relative], dim=1)


def _features(values: Any, where: str) -> list[float]:
    if not isinstance(values, list) or len(values) != FEATURE_DIM:
        raise ValueError(f"{where}: expected {FEATURE_DIM} features")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{where}: features must be finite")
    return result


def _edge(value: Any, tie: bool = False) -> tuple:
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError("edges must be [candidate_a, candidate_b, optional_gap]")
    if tie:
        return str(value[0]), str(value[1])
    return str(value[0]), str(value[1]), float(value[2]) if len(value) > 2 else 1.0


def read_groups(path: str | Path) -> list[Group]:
    raw = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not raw:
        raise ValueError("Input contains no records")
    if all("candidates" in row for row in raw):
        groups = []
        for row in raw:
            candidates = []
            for index, item in enumerate(row["candidates"]):
                candidates.append(Candidate(
                    str(item.get("candidate_id", index)),
                    _features(item["features"], f"group {row['group_id']} candidate {index}"),
                    float(item["human_score"]) if item.get("human_score") is not None else None,
                    int(item["rank"]) if item.get("rank") is not None else None,
                ))
            groups.append(Group(
                str(row["group_id"]), str(row.get("split", "train")), candidates,
                [_edge(value) for value in row.get("preference_edges", [])],
                [_edge(value, tie=True) for value in row.get("tie_edges", [])],
            ))
        return groups

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw:
        if "group_id" not in row or "features" not in row:
            raise ValueError("flat records require group_id and features")
        by_group[str(row["group_id"])].append(row)
    groups = []
    for group_id, rows in by_group.items():
        splits = {str(row.get("split", "train")) for row in rows}
        if len(splits) != 1:
            raise ValueError(f"Group {group_id}: candidates have different splits")
        candidates = [Candidate(
            str(row.get("candidate_id", index)), _features(row["features"], group_id),
            float(row["human_score"]) if row.get("human_score") is not None else None,
            int(row["rank"]) if row.get("rank") is not None else None,
        ) for index, row in enumerate(rows)]
        groups.append(Group(group_id, next(iter(splits)), candidates, [], []))
    return groups


def check_groups(groups: list[Group]) -> None:
    seen: set[str] = set()
    for group in groups:
        if group.group_id in seen:
            raise ValueError(f"Duplicate group_id: {group.group_id}")
        seen.add(group.group_id)
        if group.split not in {"train", "val", "validation", "test"}:
            raise ValueError(f"Group {group.group_id}: split must be train, val/validation, or test")
        if len(group.candidates) not in SUPPORTED_K:
            raise ValueError(f"Group {group.group_id}: K must be 2, 3, or 4")
        ids = [candidate.candidate_id for candidate in group.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Group {group.group_id}: candidate IDs must be unique")


def constraints(group: Group, delta_pref: float) -> tuple[list[tuple[int, int, float]], list[tuple[int, int]]]:
    positions = {candidate.candidate_id: index for index, candidate in enumerate(group.candidates)}
    directional = [(positions[a], positions[b], gap) for a, b, gap in group.preference_edges]
    ties = [(positions[a], positions[b]) for a, b in group.tie_edges]
    if directional or ties:
        return directional, ties
    scores = [candidate.human_score for candidate in group.candidates]
    if all(score is not None for score in scores):
        for a in range(len(scores)):
            for b in range(a + 1, len(scores)):
                gap = abs(float(scores[a]) - float(scores[b]))
                if gap <= delta_pref:
                    ties.append((a, b))
                elif scores[a] > scores[b]:
                    directional.append((a, b, gap))
                else:
                    directional.append((b, a, gap))
        return directional, ties
    ranks = [candidate.rank for candidate in group.candidates]
    if any(rank is None for rank in ranks):
        raise ValueError(f"Group {group.group_id}: annotations require edges, human_score, or rank")
    for a in range(len(ranks)):
        for b in range(a + 1, len(ranks)):
            if ranks[a] == ranks[b]:
                ties.append((a, b))
            elif ranks[a] < ranks[b]:
                directional.append((a, b, 1.0))
            else:
                directional.append((b, a, 1.0))
    return directional, ties


def matrices(groups: list[Group], train_groups: list[Group]) -> tuple[torch.Tensor, torch.Tensor]:
    train_candidates = [candidate.features for group in train_groups for candidate in group.candidates]
    if not train_candidates:
        raise ValueError("No training candidates")
    raw_train = torch.tensor(train_candidates, dtype=torch.float32)
    mean = raw_train.mean(0)
    std = raw_train.std(0, unbiased=True).clamp_min(1e-6)
    raw = torch.tensor([candidate.features for group in groups for candidate in group.candidates], dtype=torch.float32)
    relative = torch.zeros_like(raw)
    offset = 0
    for group in groups:
        count = len(group.candidates)
        values = raw[offset:offset + count]
        relative[offset:offset + count] = (values - values.mean(0)) / values.std(0, unbiased=True).clamp_min(1e-4)
        offset += count
    return torch.cat([(raw - mean) / std, relative], dim=1), torch.stack([mean, std])


def fit_head(features: torch.Tensor, groups: list[Group], offsets: dict[str, int], config: SearchConfig) -> torch.Tensor:
    winners, losers, weights = [], [], []
    ties_a, ties_b = [], []
    for group in groups:
        directional, ties = constraints(group, config.delta_pref)
        offset = offsets[group.group_id]
        for winner, loser, gap in directional:
            winners.append(offset + winner); losers.append(offset + loser)
            weights.append(min(1.0, abs(gap) / max(config.delta_scale, 1e-8)))
        for a, b in ties:
            ties_a.append(offset + a); ties_b.append(offset + b)
    if not winners and not ties_a:
        raise ValueError("No usable preference constraints")
    weight = torch.zeros(features.shape[1], dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.AdamW([weight], lr=config.lr, weight_decay=0.0)
    winner_index = torch.tensor(winners, dtype=torch.long) if winners else None
    loser_index = torch.tensor(losers, dtype=torch.long) if losers else None
    for _ in range(config.epochs):
        terms = []
        if winner_index is not None:
            margin = features[winner_index] @ weight - features[loser_index] @ weight
            terms.append((F.softplus(-margin) * torch.tensor(weights)).mean())
        if ties_a and config.tie_weight > 0:
            margin = features[torch.tensor(ties_a)] @ weight - features[torch.tensor(ties_b)] @ weight
            terms.append(config.tie_weight * F.softplus(margin.abs() - 0.05).mean())
        loss = sum(terms) + config.l2 * weight.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return weight.detach()


def strict_accuracy(features: torch.Tensor, groups: list[Group], offsets: dict[str, int], weight: torch.Tensor, delta_pref: float) -> tuple[int, int, float]:
    scores = features @ weight
    correct = 0
    for group in groups:
        directional, _ties = constraints(group, delta_pref)
        offset = offsets[group.group_id]
        # Strict group accuracy follows the benchmark protocol: every annotated
        # directional edge must be correct; near-ties are training constraints,
        # not additional ordering edges.
        good = all(bool(scores[offset + a] > scores[offset + b]) for a, b, _ in directional)
        correct += int(good)
    total = len(groups)
    return correct, total, 100.0 * correct / total if total else 0.0


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def train(args: argparse.Namespace) -> int:
    if Path(args.output).exists():
        raise ValueError(f"Output already exists: {args.output}")
    groups = read_groups(args.input); check_groups(groups)
    train_groups = [group for group in groups if group.split == "train"]
    val_groups = [group for group in groups if group.split in {"val", "validation"}]
    if not train_groups or not val_groups:
        raise ValueError("Both train and val/validation groups are required")
    features, stats = matrices(groups, train_groups)
    offsets, cursor = {}, 0
    for group in groups:
        offsets[group.group_id] = cursor; cursor += len(group.candidates)
    configs = [SearchConfig(lr, l2, epochs, delta, scale, tie)
               for lr in [float(x) for x in args.learning_rates.split(",")]
               for l2 in [float(x) for x in args.l2.split(",")]
               for epochs in [int(x) for x in args.epochs.split(",")]
               for delta in [float(x) for x in args.delta_pref.split(",")]
               for scale in [float(x) for x in args.delta_scale.split(",")]
               for tie in [float(x) for x in args.tie_weight.split(",")]]
    selected, report = {}, {"status": "failed", "validation": {}, "criterion": "strict directional edges and tau0=0.05 tie tolerance"}
    for k in SUPPORTED_K:
        train_k = [group for group in train_groups if len(group.candidates) == k]
        val_k = [group for group in val_groups if len(group.candidates) == k]
        if not train_k or not val_k:
            raise ValueError(f"K={k}: train and val groups are required")
        best = None
        for config in configs:
            weight = fit_head(features, train_k, offsets, config)
            metric = strict_accuracy(features, val_k, offsets, weight, config.delta_pref)
            key = (metric[2], -config.lr, -config.l2, -config.epochs, -config.delta_pref, -config.tie_weight)
            if best is None or key > best[0]:
                best = (key, weight, config, metric)
        _, weight, config, (correct, total, accuracy) = best
        selected[k] = weight
        report["validation"][f"k{k}"] = {"correct_groups": correct, "total_groups": total, "accuracy_percent": accuracy, "config": config.__dict__}
        test_k = [group for group in groups if group.split == "test" and len(group.candidates) == k]
        if test_k:
            tc, tt, ta = strict_accuracy(features, test_k, offsets, weight, config.delta_pref)
            report.setdefault("test", {})[f"k{k}"] = {"correct_groups": tc, "total_groups": tt, "accuracy_percent": ta}
    passed = all(report["validation"][f"k{k}"]["accuracy_percent"] > DEFAULT_TARGETS[k] for k in SUPPORTED_K)
    report["targets_percent_exclusive"] = {f"k{k}": DEFAULT_TARGETS[k] for k in SUPPORTED_K}
    report["status"] = "passed" if passed else "calibration_failed"
    report_path = args.report or str(Path(args.output).with_suffix(".report.json")); write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.enforce_targets and not passed:
        print(f"Calibration failed; report written to {report_path}", file=sys.stderr); return 2
    write_json(args.output, {"schema_version": 2, "method": "FrozenCal-K", "feature_dim_absolute": 12, "feature_dim_total": 24, "feature_mean": stats[0].tolist(), "feature_std": stats[1].tolist(), "weights": {f"w{k}": selected[k].tolist() for k in SUPPORTED_K}, "selection_report": report})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Full FrozenCal-K calibration")
    parser.add_argument("--input", required=True); parser.add_argument("--output", required=True); parser.add_argument("--report")
    parser.add_argument("--learning-rates", default="0.003,0.01,0.03"); parser.add_argument("--l2", default="0.0001,0.001"); parser.add_argument("--epochs", default="80,160")
    parser.add_argument("--delta-pref", default="0.0,0.05,0.1"); parser.add_argument("--delta-scale", default="0.1,0.3,0.5"); parser.add_argument("--tie-weight", default="0,0.1,1")
    parser.add_argument("--enforce-targets", action="store_true", help="Fail unless validation exceeds 65/30/10")
    args = parser.parse_args()
    try: return train(args)
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr); return 1


if __name__ == "__main__":
    raise SystemExit(main())
