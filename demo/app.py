"""Image -> frozen Qwen/SigLIP features -> FrozenCal-K ranking demo."""
from functools import lru_cache
from pathlib import Path
import json
import os
import tempfile

import gradio as gr
import numpy as np
import torch

from frozencal.embeddings import extract_embeddings, align_group_embeddings
from frozencal.features import candidate_features

ROOT = Path(__file__).parent
PAYLOAD = json.loads((ROOT / "weights.json").read_text())
METHOD = PAYLOAD["methods"]["FrozenCal-K"]


@lru_cache(maxsize=1)
def model_paths():
    """Use configured paths, or lazily download the public ModelScope models."""
    qwen = os.getenv("FROZENCALK_QWEN_MODEL")
    siglip = os.getenv("FROZENCALK_SIGLIP_MODEL")
    if not qwen or not siglip:
        from modelscope import snapshot_download
        qwen = qwen or snapshot_download("Qwen/Qwen3-VL-Embedding-2B")
        siglip = siglip or snapshot_download("AI-ModelScope/siglip2-base-patch16-224")
    repo = os.getenv("FROZENCALK_QWEN_REPO", "/opt/Qwen3-VL-Embedding")
    return qwen, siglip, repo


def score_images(source, instruction, edited_a, edited_b, edited_c, edited_d, k):
    images = [edited_a, edited_b, edited_c, edited_d][: int(k)]
    if source is None or not instruction.strip() or any(item is None for item in images):
        return "Upload one source image, an instruction, and exactly K edited images.", []
    with tempfile.TemporaryDirectory() as tmp:
        records = []
        for index, edited in enumerate(images):
            records.append({"group_id": "demo", "source": source, "instruction": instruction, "edited": edited, "candidate_id": chr(65 + index)})
        try:
            qwen, siglip, repo = model_paths()
            tensors, _ = extract_embeddings(records, qwen_model_path=qwen, siglip_model_dir=siglip, qwen_repo_root=repo, batch_size=1)
            aligned = align_group_embeddings(records, tensors)
            absolute = candidate_features(aligned["qwen_source"], aligned["qwen_text"], aligned["qwen_fused"], aligned["qwen_edited"], aligned["siglip_source"], aligned["siglip_text"], aligned["siglip_edited"])
        except Exception as error:
            return f"Model loading or feature extraction failed: `{error}`", []
    mean = torch.tensor(PAYLOAD["feature_mean"], dtype=torch.float32)
    std = torch.tensor(PAYLOAD["feature_std"], dtype=torch.float32).clamp_min(1e-6)
    absolute = (absolute - mean) / std
    relative = (absolute - absolute.mean(0)) / absolute.std(0).clamp_min(1e-4)
    features = torch.cat([absolute, relative], dim=1)
    weights = torch.tensor(METHOD["weights"][f"w{int(k)}"], dtype=torch.float32)
    values = features @ weights
    order = torch.argsort(values, descending=True).tolist()
    rows = [[chr(65 + i), float(values[i]), int(order.index(i) + 1)] for i in range(int(k))]
    summary = "\n".join(f"**Rank {rank}: Candidate {chr(65 + index)}** | score `{values[index]:.4f}`" for rank, index in enumerate(order, 1))
    return summary, rows


with gr.Blocks(title="FrozenCal-K Image Editor Evaluator") as app:
    gr.Markdown("# FrozenCal-K image-edit evaluator\nUpload a source image, an editing instruction, and K edited candidates. The app extracts the frozen QwenVL/SigLIP2 features and applies the released K-specific calibration head.")
    source = gr.Image(type="filepath", label="Source image")
    instruction = gr.Textbox(label="Editing instruction", placeholder="e.g. make the sky sunset orange")
    k = gr.Radio([2, 3, 4], value=2, label="Number of candidates (K)")
    with gr.Row():
        edited_a = gr.Image(type="filepath", label="Candidate A")
        edited_b = gr.Image(type="filepath", label="Candidate B")
        edited_c = gr.Image(type="filepath", label="Candidate C")
        edited_d = gr.Image(type="filepath", label="Candidate D")
    run = gr.Button("Extract features and score", variant="primary")
    result = gr.Markdown()
    scores = gr.Dataframe(headers=["Candidate", "FrozenCal-K score", "Rank"], datatype=["str", "number", "number"], label="Ranking")
    run.click(score_images, [source, instruction, edited_a, edited_b, edited_c, edited_d, k], [result, scores])

if __name__ == "__main__":
    app.launch()
