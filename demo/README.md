# FrozenCal-K online demo

This Gradio app accepts a source image, an editing instruction, and 2--4 edited
candidate images. It extracts frozen QwenVL-Embedding-2B and SigLIP2 features,
then applies the released K-specific FrozenCal-K head. ModelScope downloads the
two public encoder checkpoints on first use; the first request can take several
minutes. The Jupyter notebook demonstrates the same scoring head locally.

Run locally with `pip install -r requirements.txt && python app.py`.
