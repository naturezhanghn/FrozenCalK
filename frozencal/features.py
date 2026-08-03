from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import torch
import torch.nn.functional as F


ABS_FEATURE_NAMES = [
    "qwen_fused_to_edit",
    "qwen_text_to_edit",
    "qwen_source_to_edit",
    "qwen_residual_direction",
    "qwen_edit_magnitude",
    "qwen_text_gain",
    "siglip2_text_to_edit",
    "siglip2_source_to_edit",
    "siglip2_text_gain",
    "qwen_siglip_agreement",
    "qwen_over_edit",
    "qwen_under_edit",
]

REL_FEATURE_NAMES = [f"rel_{name}" for name in ABS_FEATURE_NAMES]


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity for a batch of embeddings."""
    return (F.normalize(a.float(), dim=-1) * F.normalize(b.float(), dim=-1)).sum(dim=-1)


def candidate_features(
    qwen_source: torch.Tensor,
    qwen_text: torch.Tensor,
    qwen_fused: torch.Tensor,
    qwen_edit: torch.Tensor,
    siglip_source: torch.Tensor,
    siglip_text: torch.Tensor,
    siglip_edit: torch.Tensor,
) -> torch.Tensor:
    """Build the 12 absolute FrozenCal features for edited-image candidates.

    Inputs are batched embeddings aligned row-wise. Qwen embeddings provide
    source-conditioned residual geometry; SigLIP2 embeddings provide an
    independent text-image and source-image alignment space.
    """
    q_src = F.normalize(qwen_source.float(), dim=-1)
    q_txt = F.normalize(qwen_text.float(), dim=-1)
    q_fused = F.normalize(qwen_fused.float(), dim=-1)
    q_edit = F.normalize(qwen_edit.float(), dim=-1)
    s_src = F.normalize(siglip_source.float(), dim=-1)
    s_txt = F.normalize(siglip_text.float(), dim=-1)
    s_edit = F.normalize(siglip_edit.float(), dim=-1)

    q_fused_to_edit = cosine(q_fused, q_edit)
    q_text_to_edit = cosine(q_txt, q_edit)
    q_source_to_edit = cosine(q_src, q_edit)
    q_residual_direction = cosine(q_edit - q_src, q_fused - q_src)
    q_edit_magnitude = torch.log(
        (1.0 - q_source_to_edit).clamp_min(1e-6)
        / (1.0 - cosine(q_src, q_fused)).clamp_min(1e-6)
    )
    q_text_gain = q_text_to_edit - cosine(q_txt, q_src)

    siglip_text_to_edit = cosine(s_txt, s_edit)
    siglip_source_to_edit = cosine(s_src, s_edit)
    siglip_text_gain = siglip_text_to_edit - cosine(s_txt, s_src)

    agreement = q_text_gain * siglip_text_gain
    over_edit = torch.relu(q_edit_magnitude)
    under_edit = torch.relu(-q_edit_magnitude)

    return torch.stack(
        [
            q_fused_to_edit,
            q_text_to_edit,
            q_source_to_edit,
            q_residual_direction,
            q_edit_magnitude,
            q_text_gain,
            siglip_text_to_edit,
            siglip_source_to_edit,
            siglip_text_gain,
            agreement,
            over_edit,
            under_edit,
        ],
        dim=1,
    )


def standardize_features(
    features: torch.Tensor,
    mean: Iterable[float] | torch.Tensor,
    std: Iterable[float] | torch.Tensor,
) -> torch.Tensor:
    """Apply release-time feature standardization."""
    mean_t = torch.as_tensor(mean, dtype=features.dtype, device=features.device)
    std_t = torch.as_tensor(std, dtype=features.dtype, device=features.device).clamp_min(1e-6)
    return (features - mean_t) / std_t


def group_relative_features(features: torch.Tensor, group_ids: Iterable[str | int]) -> torch.Tensor:
    """Compute within-group z-scored relative features.

    Candidate-set-relative evidence is computed per source-instruction group.
    Groups with a single candidate receive zero relative features.
    """
    rel = torch.zeros_like(features.float())
    groups: dict[str | int, list[int]] = defaultdict(list)
    for i, group_id in enumerate(group_ids):
        groups[group_id].append(i)
    for indices in groups.values():
        if len(indices) <= 1:
            continue
        idx = torch.tensor(indices, dtype=torch.long, device=features.device)
        vals = features[idx].float()
        rel[idx] = (vals - vals.mean(0)) / vals.std(0).clamp_min(1e-4)
    return rel


def abs_rel_features(
    abs_features: torch.Tensor,
    group_ids: Iterable[str | int],
    mean: Iterable[float] | torch.Tensor,
    std: Iterable[float] | torch.Tensor,
) -> torch.Tensor:
    """Return the 24-dim FrozenCal abs/rel feature matrix."""
    return torch.cat(
        [
            standardize_features(abs_features, mean, std),
            group_relative_features(abs_features, group_ids),
        ],
        dim=1,
    )
