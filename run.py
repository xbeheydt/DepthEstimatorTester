"""
Gradio App entrypoint.
"""

import argparse
import sys

import gradio as gr


class App:
    """
    Class handles a Gradio app.
    """

    TITLE = "# Depth Estimator Model Tester."
    DESCRIPTION = "Gradio app for testing **Depth estimator** AI models."
    MODEL_ZOO = {
        "LiheYoung/depth-anything": ["small", "base", "large"],
    }
    CMAP = ["magma_r", "gray_r"]

    def __init__(self) -> None:
        self.cliarg()

        with gr.Blocks() as self.demo:
            self.header()

            with gr.Row():
                with gr.Column():
                    self.settings()
                with gr.Column():
                    self.input()

            self.outputs()
            self.callbacks()

    def cliarg(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--public",
            default=False,
            help="Turn on public mode.",
            action="store_true",
        )
        self.__args = parser.parse_args()

    def header(self) -> None:
        gr.Markdown(self.TITLE)
        gr.Markdown(self.DESCRIPTION)

    def settings(self) -> None:
        self.model_name = gr.Dropdown(
            label="Model",
            choices=self.MODEL_ZOO.keys(),
        )
        self.model_size = gr.Dropdown(label="Size", choices=[])
        self.cmap = gr.Radio(
            label="Depth Colormap",
            choices=self.CMAP,
            value=self.CMAP[0],
        )
        self.keep_edges = gr.Checkbox(
            label="Keep Oclusion Edges",
            value=False,
        )

    def input(self) -> None:
        self.input_image = gr.Image(
            label="Image Input",
            type="pil",
        )

    def outputs(self) -> None:
        self.depth_image = gr.Image(label="Depth Map")
        self.raw_file = gr.File(label="16-bit raw depth, multiplier:256")
        self.result_3d = gr.Model3D(
            label="3d mesh reconstruction",
        )

    def callbacks(self)-> None:
        self.model_name.change(self.cb_model_name, [])

    def run(self) -> None:
        self.demo.launch(share=self.__args.public)


if __name__ == "__main__":
    app = App()
    sys.exit(app.run())
