"""
Entrypoint of Gradio app

This app runs a gradio interface for testing Depth estimator AI models.
"""

import torch
import gradio as gr
import sys
import argparse

title = "# Depth Estimator Model Tester"
description = "Gradio app for testing **Depth estimator** AI models."
device = "cuda" if torch.cuda.is_available() else "cpu"


def cliargs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--public",
        type=str,
        default=False,
        help="Turn on public mode.",
        type=bool,
    )
    return parser.parse_args()


# App structure ===============================================================
with gr.Blocks() as app:
    gr.Markdown(title)
    gr.Markdown(description)
# ! App structure =============================================================

if __name__ == "__main__":
    args = cliargs()
    sys.exit(app.queue().launch(share=args.public))
