"""
Entrypoint of Gradio app

This app runs a gradio interface for testing Depth estimator AI models.
"""

from PIL import Image
import argparse
import gradio as gr
import matplotlib
import numpy as np
import shutil
import sys
import tempfile
import torch


TITLE = "# Depth Estimator Model Tester"
DESCRIPTION = "Gradio app for testing **Depth estimator** AI models."
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_OPTIONS = {"isl-org/ZoeDepth": ["ZoeD_N", "ZoeD_K", "ZoeD_NK"]}
MODEL = None  # Unloaded model at start
CMAP_OPTIONS = ["magma_r", "gray_r"]


def cliargs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--public",
        default=False,
        help="Turn on public mode.",
        action="store_true",
    )
    return parser.parse_args()


def infer(model, img):
    return model.infer_pil(img)


def colorize(
    value,
    vmin=None,
    vmax=None,
    cmap="magma_r",
    invalid_val=-99,
    invalid_mask=None,
    background_color=(128, 128, 128, 255),
    gamma_corrected=False,
    value_transform=None,
):
    """Converts a depth map to a color image.
    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.
    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask], 2) if vmin is None else vmin
    vmax = np.percentile(value[mask], 85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.0

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img


# App structure ===============================================================
with gr.Blocks() as app:
    gr.Markdown(TITLE)
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column():
            model_name = gr.Dropdown(
                label="Model",
                choices=MODEL_OPTIONS.keys(),
            )
            model_size = gr.Dropdown(
                label="Size",
                choices=[],
            )
            cmap = gr.Radio(
                label="Colormap Depth",
                choices=CMAP_OPTIONS,
                value=CMAP_OPTIONS[0],
            )
            occ_edges = gr.Checkbox(label="Keep occlusion edges", value=False)
        with gr.Column():
            input_image = gr.Image(label="Input Image", type="pil")

    depth_image = gr.Image(label="Depth Map")
    raw_file = gr.File(label="16-bit raw depth, multiplier:256")
    result_3d = gr.Model3D(
        label="3d mesh reconstruction",
        clear_color=[1.0, 1.0, 1.0, 1.0],
    )
    execute = gr.Button("Execute", interactive=True)
# ! App structure =============================================================

# Element Callbacks ===========================================================
    def choose_model(name):
        return gr.Dropdown(choices=MODEL_OPTIONS[name])

    model_name.change(choose_model, model_name, [model_size])

    def choose_model_size(name, size):
        shutil.rmtree(torch.hub.get_dir())
        global MODEL
        MODEL = (
            torch.hub.load(
                name,
                size,
                pretained=True,
            )
            .to(DEVICE)
            .eval()
        )
        return gr.Button(interactive=True)

    model_size.change(
        choose_model_size,
        [model_name, model_size],
        [execute],
        show_progress=True,
    )

    def on_execute(image, cmap):
        global MODEL
        depth = infer(MODEL, image)
        colored_depth = colorize(depth, cmap=cmap)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        raw_depth = Image.fromarray((depth * 256).astype("uint16"))
        raw_depth.save(tmp.name)
        return [colored_depth, tmp.name]

    execute.click(
        on_execute,
        inputs=[input_image],
        outputs=[depth_image, raw_file],
    )

# ! Element Callbacks =========================================================

if __name__ == "__main__":
    args = cliargs()
    sys.exit(app.queue().launch(share=args.public))
