"""Small ModelScope/Gradio playground for the released FrozenCal-K heads."""
from pathlib import Path
import json
import numpy as np
import gradio as gr

ROOT = Path(__file__).parent
PAYLOAD = json.loads((ROOT / "weights.json").read_text())
METHOD = PAYLOAD["methods"]["FrozenCal-K"]
FEATURES = PAYLOAD["feature_names"][:12]

def rank_candidates(rows, k):
    if rows is None or len(rows) < 2:
        return "Please provide at least two candidate rows.", []
    values = np.asarray(rows, dtype=float)[:, :12]
    if values.shape[1] != 12 or not np.isfinite(values).all():
        return "Every candidate needs 12 finite numeric feature values.", []
    k = int(k)
    if len(values) != k:
        return f"The selected K={k} requires exactly {k} candidates.", []
    absolute = (values - values.mean(0)) / np.maximum(values.std(0), 1e-4)
    relative = absolute.copy()
    weights = np.asarray(METHOD["weights"][f"w{k}"])
    scores = np.concatenate([absolute, relative], axis=1) @ weights
    order = np.argsort(-scores)
    table = [[int(i + 1), float(scores[i]), int(np.where(order == i)[0][0] + 1)] for i in range(k)]
    text = "\n".join(f"**Rank {rank}: Candidate {idx + 1}** | score `{scores[idx]:.4f}`" for rank, idx in enumerate(order, 1))
    return text, table

demo_rows = [[0.72, 0.35, 0.76, 0.42, -0.62, 0.09, 0.11, 0.87, 0.02, 0.01, 0.09, 0.72] for _ in range(4)]
demo_rows[1][2] += 0.08
demo_rows[2][4] -= 0.10
demo_rows[3][7] += 0.12

with gr.Blocks(title="FrozenCal-K Playground") as app:
    gr.Markdown("# FrozenCal-K\nInteractive ranking with frozen QwenVL/SigLIP2 features. The encoders remain frozen; only the released K-specific linear head is applied.")
    with gr.Row():
        k = gr.Radio([2, 3, 4], value=4, label="Number of candidates (K)")
        run = gr.Button("Rank candidates", variant="primary")
    gr.Markdown("Enter one candidate per row. Columns are the 12 absolute features: `" + "`, `".join(FEATURES) + "`.")
    table = gr.Dataframe(value=demo_rows, headers=FEATURES, datatype="number", row_count=(4, "dynamic"), col_count=(12, "fixed"), label="Candidate features")
    result = gr.Markdown()
    scores = gr.Dataframe(headers=["Candidate", "Raw score", "Rank"], datatype=["number", "number", "number"], label="Scores")
    run.click(rank_candidates, inputs=[table, k], outputs=[result, scores])

if __name__ == "__main__":
    app.launch()
