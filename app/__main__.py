"""
Entrypoint of Gradio app

This app runs a gradio interface for testing Depth estimator AI models.
"""

import torch
import gradio as gr
import sys

title = "# Depth Estimator Model Tester"
description = "Gradio app for testing **Depth estimator** AI models."
device = "cuda" if torch.cuda.is_available() else "cpu"

# App structure ===============================================================
with gr.Blocks() as app:
    gr.Markdown(title)
    gr.Markdown(description)
# ! App structure =============================================================

if __name__ == "__main__":
    sys.exit(app.queue().launch())
