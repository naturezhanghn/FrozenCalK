from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from .io import unique_groups


FUSED_INSTRUCTION = (
    "Given a source image and an editing instruction, represent the desired edited image. "
    "The representation should focus on the visual result after applying the instruction, "
    "and should preserve all irrelevant details from the source image."
)

TEXT_INSTRUCTION = "Represent the image editing instruction."


def _load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _save_index(cache_dir: Path, records: list[dict[str, Any]], groups: list[dict[str, Any]]) -> None:
    payload = {
        "num_records": len(records),
        "num_groups": len(groups),
        "groups": [
            {
                "group_id": row["group_id"],
                "source": row["source"],
                "instruction": row["instruction"],
            }
            for row in groups
        ],
        "records": [
            {
                "group_id": row["group_id"],
                "edited": row["edited"],
                "candidate_id": row.get("candidate_id", ""),
                "model": row.get("model", ""),
            }
            for row in records
        ],
    }
    (cache_dir / "index.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_cached_embeddings(cache_dir: str | Path) -> dict[str, torch.Tensor]:
    cache_dir = Path(cache_dir)
    names = [
        "qwen_source",
        "qwen_text",
        "qwen_fused",
        "qwen_edited",
        "siglip_source",
        "siglip_text",
        "siglip_edited",
    ]
    missing = [name for name in names if not (cache_dir / f"{name}.pt").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing cached embedding files in {cache_dir}: {missing}")
    return {name: torch.load(cache_dir / f"{name}.pt", map_location="cpu").float() for name in names}


def save_cached_embeddings(cache_dir: str | Path, tensors: dict[str, torch.Tensor], records: list[dict[str, Any]]) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, value in tensors.items():
        torch.save(value.half().cpu(), cache_dir / f"{name}.pt")
    _save_index(cache_dir, records, unique_groups(records))


def load_qwen_embedder(model_path: str | Path, repo_root: str | Path | None = None, *, dtype: str = "float16"):
    if repo_root is not None and str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

    # CPU backends are more reliable with float32; float16 is reserved for CUDA.
    torch_dtype = torch.float16 if dtype == "float16" and torch.cuda.is_available() else torch.float32
    return Qwen3VLEmbedder(
        model_name_or_path=str(model_path),
        max_length=512,
        min_pixels=4 * 32 * 32,
        max_pixels=512 * 512,
        torch_dtype=torch_dtype,
    )


@torch.no_grad()
def embed_qwen(
    records: list[dict[str, Any]],
    model_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    batch_size: int = 8,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    embedder = load_qwen_embedder(model_path, repo_root=repo_root)
    groups = unique_groups(records)

    def process(payloads: list[dict[str, Any]]) -> torch.Tensor:
        chunks = []
        for start in range(0, len(payloads), batch_size):
            chunks.append(embedder.process(payloads[start : start + batch_size], normalize=True).float().cpu())
        return torch.cat(chunks, dim=0)

    tensors = {
        "qwen_source": process([{"image": row["source"]} for row in groups]),
        "qwen_text": process(
            [{"text": row["instruction"], "instruction": TEXT_INSTRUCTION} for row in groups]
        ),
        "qwen_fused": process(
            [
                {"image": row["source"], "text": row["instruction"], "instruction": FUSED_INSTRUCTION}
                for row in groups
            ]
        ),
        "qwen_edited": process([{"image": row["edited"]} for row in records]),
    }
    return tensors, groups


@torch.no_grad()
def embed_siglip2(
    records: list[dict[str, Any]],
    model_dir: str | Path,
    *,
    batch_size: int = 16,
    device: str | None = None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    from transformers import AutoModel, AutoProcessor

    device_t = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True).to(device_t).eval()
    groups = unique_groups(records)

    def as_tensor(output: Any) -> torch.Tensor:
        if torch.is_tensor(output):
            return output
        for attr in ("image_embeds", "text_embeds", "pooler_output"):
            value = getattr(output, attr, None)
            if torch.is_tensor(value):
                return value
        if isinstance(output, (tuple, list)):
            for value in reversed(output):
                if torch.is_tensor(value) and value.ndim == 2:
                    return value
        raise TypeError(f"Cannot extract an embedding tensor from {type(output)!r}")

    def image_features(paths: list[str]) -> torch.Tensor:
        out = []
        for start in range(0, len(paths), batch_size):
            chunk = paths[start : start + batch_size]
            inputs = processor(images=[_load_rgb(path) for path in chunk], return_tensors="pt").to(device_t)
            out.append(F.normalize(as_tensor(model.get_image_features(**inputs)).float(), dim=-1).cpu())
        return torch.cat(out, dim=0)

    def text_features(texts: list[str]) -> torch.Tensor:
        out = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            inputs = processor(
                text=chunk,
                padding="max_length",
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(device_t)
            out.append(F.normalize(as_tensor(model.get_text_features(**inputs)).float(), dim=-1).cpu())
        return torch.cat(out, dim=0)

    tensors = {
        "siglip_source": image_features([row["source"] for row in groups]),
        "siglip_text": text_features([row["instruction"] for row in groups]),
        "siglip_edited": image_features([row["edited"] for row in records]),
    }
    return tensors, groups


def extract_embeddings(
    records: list[dict[str, Any]],
    *,
    qwen_model_path: str | Path,
    siglip_model_dir: str | Path,
    qwen_repo_root: str | Path | None = None,
    batch_size: int = 8,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    qwen, qwen_groups = embed_qwen(records, qwen_model_path, repo_root=qwen_repo_root, batch_size=batch_size)
    timings["qwen_seconds"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    siglip, siglip_groups = embed_siglip2(records, siglip_model_dir, batch_size=batch_size)
    timings["siglip2_seconds"] = time.perf_counter() - t1
    if [row["group_id"] for row in qwen_groups] != [row["group_id"] for row in siglip_groups]:
        raise RuntimeError("Qwen and SigLIP2 group orders do not match")
    timings["sequential_seconds"] = timings["qwen_seconds"] + timings["siglip2_seconds"]
    timings["parallel_estimated_seconds"] = max(timings["qwen_seconds"], timings["siglip2_seconds"])
    return {**qwen, **siglip}, timings


def align_group_embeddings(records: list[dict[str, Any]], tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    groups = unique_groups(records)
    group_to_pos = {row["group_id"]: i for i, row in enumerate(groups)}
    sample_pos = torch.tensor([group_to_pos[row["group_id"]] for row in records], dtype=torch.long)
    return {
        "qwen_source": tensors["qwen_source"][sample_pos],
        "qwen_text": tensors["qwen_text"][sample_pos],
        "qwen_fused": tensors["qwen_fused"][sample_pos],
        "qwen_edited": tensors["qwen_edited"],
        "siglip_source": tensors["siglip_source"][sample_pos],
        "siglip_text": tensors["siglip_text"][sample_pos],
        "siglip_edited": tensors["siglip_edited"],
    }
