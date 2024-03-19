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
import trimesh


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


def depth_edges_mask(depth):
    """Returns a mask of edges in the depth map.
    Args:
    depth: 2D numpy array of shape (H, W) with dtype float32.
    Returns:
    mask: 2D numpy array of shape (H, W) with dtype bool.
    """
    # Compute the x and y gradients of the depth map.
    depth_dx, depth_dy = np.gradient(depth)
    # Compute the gradient magnitude.
    depth_grad = np.sqrt(depth_dx**2 + depth_dy**2)
    # Compute the edge mask.
    mask = depth_grad > 0.05
    return mask


def get_intrinsics(H, W):
    """
    Intrinsics for a pinhole camera model.
    Assume fov of 55 degrees and central principal point.
    """
    f = 0.5 * W / np.tan(0.5 * 55 * np.pi / 180.0)
    cx = 0.5 * W
    cy = 0.5 * H
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])


def depth_to_points(depth, R=None, t=None):

    K = get_intrinsics(depth.shape[1], depth.shape[2])
    Kinv = np.linalg.inv(K)
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)

    # M converts from your coordinate to PyTorch3D's coordinate system
    M = np.eye(3)
    M[0, 0] = -1.0
    M[1, 1] = -1.0

    height, width = depth.shape[1:3]

    x = np.arange(width)
    y = np.arange(height)
    coord = np.stack(np.meshgrid(x, y), -1)
    coord = np.concatenate((coord, np.ones_like(coord)[:, :, [0]]), -1)  # z=1
    coord = coord.astype(np.float32)
    # coord = torch.as_tensor(coord, dtype=torch.float32, device=device)
    coord = coord[None]  # bs, h, w, 3

    D = depth[:, :, :, None, None]
    # print(D.shape, Kinv[None, None, None, ...].shape, coord[:, :, :, :, None].shape )
    pts3D_1 = D * Kinv[None, None, None, ...] @ coord[:, :, :, :, None]
    # pts3D_1 live in your coordinate system. Convert them to Py3D's
    pts3D_1 = M[None, None, None, ...] @ pts3D_1
    # from reference to targe tviewpoint
    pts3D_2 = R[None, None, None, ...] @ pts3D_1 + t[None, None, None, :, None]
    # pts3D_2 = pts3D_1
    # depth_2 = pts3D_2[:, :, :, 2, :]  # b,1,h,w
    return pts3D_2[:, :, :, :3, 0][0]


def create_triangles(h, w, mask=None):
    """Creates mesh triangle indices from a given pixel grid size.
        This function is not and need not be differentiable as triangle indices are
        fixed.
    Args:
    h: (int) denoting the height of the image.
    w: (int) denoting the width of the image.
    Returns:
    triangles: 2D numpy array of indices (int) with shape (2(W-1)(H-1) x 3)
    """
    x, y = np.meshgrid(range(w - 1), range(h - 1))
    tl = y * w + x
    tr = y * w + x + 1
    bl = (y + 1) * w + x
    br = (y + 1) * w + x + 1
    triangles = np.array([tl, bl, tr, br, tr, bl])
    triangles = np.transpose(triangles, (1, 2, 0)).reshape(((w - 1) * (h - 1) * 2, 3))
    if mask is not None:
        mask = mask.reshape(-1)
        triangles = triangles[mask[triangles].all(1)]
    return triangles


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
            keep_edges = gr.Checkbox(label="Keep occlusion edges", value=False)
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
        shutil.rmtree(torch.hub.get_dir(), ignore_errors=True)
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

    # FIXME : using partial for send model
    model_size.change(
        choose_model_size,
        [model_name, model_size],
        [execute],
        show_progress=True,
    )

    def on_execute(image, cmap, keep_edges=False):
        global MODEL
        depth = infer(MODEL, image)
        colored_depth = colorize(depth, cmap=cmap)

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)

        raw_depth = Image.fromarray((depth * 256).astype("uint16"))
        raw_depth.save(tmp.name)

        pts3d = depth_to_points(depth[None])
        pts3d = pts3d.reshape(-1, 3)

        verts = pts3d.reshape(-1, 3)
        image = np.array(image)
        if keep_edges:
            triangles = create_triangles(image.shape[0], image.shape[1])
        else:
            triangles = create_triangles(
                image.shape[0],
                image.shape[1],
                mask=~depth_edges_mask(depth),
            )
        colors = image.reshape(-1, 3)
        mesh = trimesh.Trimesh(
            vertices=verts,
            faces=triangles,
            vertex_colors=colors,
        )

        # Save as glb
        glb_file = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
        glb_path = glb_file.name
        mesh.export(glb_path)
        return [colored_depth, tmp.name, glb_path]

    execute.click(
        on_execute,
        inputs=[input_image, cmap, keep_edges],
        outputs=[depth_image, raw_file, result_3d],
    )

# ! Element Callbacks =========================================================

if __name__ == "__main__":
    args = cliargs()
    sys.exit(app.queue().launch(share=args.public))
