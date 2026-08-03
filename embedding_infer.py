#!/usr/bin/env python3
"""Reproduce the reported FrozenCal-K metrics from cached embeddings.

This is an inference-only entry point. It reads the release embedding shards
and the released K-specific head, then evaluates the fixed paper reporting
splits for GenAI-Bench and EditReward-Bench.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from frozencal.features import candidate_features


def load_qwen_dir(path: Path) -> dict[str, Any]:
    index = json.loads((path / "index.json").read_text())
    chunks: dict[str, list[torch.Tensor]] = defaultdict(list)
    metadata: list[dict[str, Any]] = []
    for shard_id in range(int(index["num_shards"])):
        shard = torch.load(path / f"shard_{shard_id:04d}.pt", map_location="cpu")
        for key in ("src", "instr", "tgt", "fused"):
            chunks[key].append(F.normalize(shard[key].float(), dim=-1))
        metadata.extend(shard["metadata"])
    return {**{key: torch.cat(value) for key, value in chunks.items()}, "metadata": metadata}


def load_siglip_dir(path: Path, keys: list[str]) -> dict[str, Any]:
    index = json.loads((path / "index.json").read_text())
    chunks: dict[str, list[torch.Tensor]] = defaultdict(list)
    metadata: list[dict[str, Any]] = []
    for item in index["shards"]:
        shard = torch.load(path / item["path"], map_location="cpu")
        for key in keys:
            chunks[key].append(shard[key].float())
        metadata.extend(shard["metadata"])
    return {**{key: torch.cat(value) for key, value in chunks.items()}, "metadata": metadata}


def group_z(features: torch.Tensor, metadata: list[dict[str, Any]], key: str) -> torch.Tensor:
    out = torch.zeros_like(features.float())
    groups: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(metadata):
        groups[str(item[key])].append(index)
    for indices in groups.values():
        index = torch.tensor(indices, dtype=torch.long)
        values = features[index]
        out[index] = (values - values.mean(0)) / values.std(0).clamp_min(1e-4)
    return out


def split_indices(n: int, fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=generator)
    count = int(round(n * fraction))
    return order[:count], order[count:]


def pair_indices(metadata: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    by_row: dict[int, list[int]] = defaultdict(list)
    for index, item in enumerate(metadata):
        by_row[int(item["row_index"])].append(index)
    preferred, rejected = [], []
    for indices in by_row.values():
        pos = [i for i in indices if metadata[i].get("is_preferred") is True and int(metadata[i].get("num_candidates", 0)) == 2]
        neg = [i for i in indices if metadata[i].get("is_preferred") is False and int(metadata[i].get("num_candidates", 0)) == 2]
        if len(pos) == 1 and len(neg) == 1:
            preferred.append(pos[0]); rejected.append(neg[0])
    return torch.tensor(preferred), torch.tensor(rejected)


def strict_edges(metadata: list[dict[str, Any]], n_way: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(metadata):
        groups[str(item["group_id"])].append(index)
    winners, losers, group_ids = [], [], []
    group_number = 0
    for indices in groups.values():
        if int(metadata[indices[0]]["num_candidates"]) != n_way:
            continue
        letters = {str(metadata[i]["candidate_letter"]): i for i in indices}
        start = len(winners)
        for edge in metadata[indices[0]]["edges"]:
            winner = letters.get(str(edge.get("winner")))
            loser = letters.get(str(edge.get("loser")))
            if winner is not None and loser is not None:
                winners.append(winner); losers.append(loser)
        if len(winners) > start:
            group_ids.extend([group_number] * (len(winners) - start))
            group_number += 1
    return (torch.tensor(winners), torch.tensor(losers), torch.tensor(group_ids), group_number)


def load_data(data_root: Path) -> dict[str, Any]:
    emb = data_root / "embeddings"
    pair_q = load_qwen_dir(emb / "editreward_bench_pair_qwen2b")
    pair_s = load_siglip_dir(emb / "editreward_bench_pair_siglip2", ["src", "text", "tgt"])
    multi_q = load_qwen_dir(emb / "editreward_bench_multicandidate_qwen2b")
    multi_s = load_siglip_dir(emb / "editreward_bench_multicandidate_siglip2", ["src", "text", "tgt"])

    pair_abs = candidate_features(pair_q["src"], pair_q["instr"], pair_q["fused"], pair_q["tgt"], pair_s["src"], pair_s["text"], pair_s["tgt"])
    multi_abs = candidate_features(multi_q["src"], multi_q["instr"], multi_q["fused"], multi_q["tgt"], multi_s["src"], multi_s["text"], multi_s["tgt"])
    mean, std = pair_abs.mean(0), pair_abs.std(0).clamp_min(1e-6)
    pair_x = torch.cat([(pair_abs - mean) / std, group_z(pair_abs, pair_q["metadata"], "row_index")], dim=1)
    multi_x = torch.cat([(multi_abs - mean) / std, group_z(multi_abs, multi_q["metadata"], "group_id")], dim=1)

    gen_q = torch.load(emb / "genai_bench_qwen2b" / "embeddings.pt", map_location="cpu")
    gen_s = torch.load(emb / "genai_bench_siglip2" / "embeddings.pt", map_location="cpu")
    left_abs = candidate_features(gen_q["source"], gen_q["text"], gen_q["fused"], gen_q["left"], gen_s["source"], gen_s["text"], gen_s["left"])
    right_abs = candidate_features(gen_q["source"], gen_q["text"], gen_q["fused"], gen_q["right"], gen_s["source"], gen_s["text"], gen_s["right"])
    pair = torch.stack([left_abs, right_abs], dim=1)
    relative = (pair - pair.mean(1, keepdim=True)) / pair.std(1, keepdim=True).clamp_min(1e-4)
    left_x = torch.cat([(left_abs - mean) / std, relative[:, 0]], dim=1)
    right_x = torch.cat([(right_abs - mean) / std, relative[:, 1]], dim=1)
    gen_rows = [i for i, item in enumerate(gen_q["metadata"]) if item.get("split") == "test" and item.get("vote_type") in {"leftvote", "rightvote"}]
    gen_labels = torch.tensor([gen_q["metadata"][i]["vote_type"] == "leftvote" for i in gen_rows])
    return {"pair_x": pair_x, "multi_x": multi_x, "left_x": left_x, "right_x": right_x, "pref": pair_indices(pair_q["metadata"]), "e3": strict_edges(multi_q["metadata"], 3), "e4": strict_edges(multi_q["metadata"], 4), "gen_idx": torch.tensor(gen_rows), "gen_labels": gen_labels}


def group_accuracy(values: torch.Tensor, edges: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int], selected: torch.Tensor) -> tuple[int, int]:
    winners, losers, group_ids, _ = edges
    correct = 0
    for group_id in selected.tolist():
        mask = group_ids == int(group_id)
        correct += int(bool((values[winners[mask]] > values[losers[mask]]).all()))
    return correct, len(selected)


def main() -> int:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="FrozenCal-K inference from cached GenAI/ERB embeddings")
    parser.add_argument("--data-root", default=str(project_root / "reproduction_data"))
    parser.add_argument("--weights", default=str(project_root / "reproduction_data/weights/frozencal_2b.json"))
    parser.add_argument("--output-dir", default=str(project_root / "reproduced_results/embedding_inference"))
    args = parser.parse_args()
    data = load_data(Path(args.data_root))
    payload = json.loads(Path(args.weights).read_text())
    raw = payload["methods"]["FrozenCal-K"]["weights"]
    weights = {key: torch.tensor(raw[key], dtype=torch.float32) for key in ("w2", "w3", "w4")}
    gen_cal, gen_test = split_indices(len(data["gen_idx"]), 0.3, 20260728)
    k2_cal, k2_test = split_indices(len(data["pref"][0]), 0.3, 20260729)
    k3_cal, k3_test = split_indices(data["e3"][3], 0.3, 20260730)
    k4_cal, k4_test = split_indices(data["e4"][3], 0.3, 20260731)
    del gen_cal, k2_cal, k3_cal, k4_cal
    w2 = data["pair_x"] @ weights["w2"]
    gen_pos = data["gen_idx"][gen_test]
    gen_ok = ((data["left_x"][gen_pos] @ weights["w2"] - data["right_x"][gen_pos] @ weights["w2"]) > 0) == data["gen_labels"][gen_test]
    k2_ok = w2[data["pref"][0][k2_test]] > w2[data["pref"][1][k2_test]]
    k3 = group_accuracy(data["multi_x"] @ weights["w3"], data["e3"], k3_test)
    k4 = group_accuracy(data["multi_x"] @ weights["w4"], data["e4"], k4_test)
    rows = [
        {"method": "FrozenCal-K", "genai": 100 * int(gen_ok.sum()) / len(gen_ok), "erb2": 100 * int(k2_ok.sum()) / len(k2_ok), "erb3": 100 * k3[0] / k3[1], "erb4": 100 * k4[0] / k4[1]},
    ]
    rows[0]["erb_all"] = (rows[0]["erb2"] + rows[0]["erb3"] + rows[0]["erb4"]) / 3
    rows[0].update({"genai_correct": int(gen_ok.sum()), "genai_total": len(gen_ok), "erb2_correct": int(k2_ok.sum()), "erb2_total": len(k2_ok), "erb3_correct": k3[0], "erb3_total": k3[1], "erb4_correct": k4[0], "erb4_total": k4[1]})
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(rows[0], indent=2) + "\n")
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(rows[0], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
