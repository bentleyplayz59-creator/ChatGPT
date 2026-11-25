import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

import base64
import gc
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import spaces
import torch
import open3d as o3d
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
from rembg import remove, new_session
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

MODEL = None
REMBG_SESSION = None
DEVICE = "cuda"

EMBEDDED_IMAGES_B64 = [
    "<PLACEHOLDER: paste full base64 of image 1 here (see image_1_base64.txt)>",
    "<PLACEHOLDER: paste full base64 of image 2 here (see image_2_base64.txt)>",
    "<PLACEHOLDER: paste full base64 of image 3 here (see image_3_base64.txt)>",
    "<PLACEHOLDER: paste full base64 of image 4 here (see image_4_base64.txt)>",
    "<PLACEHOLDER: paste full base64 of image 5 here (see image_5_base64.txt)>",
]


def get_models():
    global MODEL, REMBG_SESSION
    if MODEL is None:
        model = VGGT()
        weights_url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        state = torch.hub.load_state_dict_from_url(
            weights_url, map_location="cpu", progress=True
        )
        model.load_state_dict(state)
        model.eval()
        MODEL = model.to(DEVICE)
    if REMBG_SESSION is None:
        REMBG_SESSION = new_session("u2net")
    return MODEL, REMBG_SESSION


def write_embedded_images(workdir):
    paths = []
    for index, b64 in enumerate(EMBEDDED_IMAGES_B64):
        path = str(Path(workdir) / f"source_{index:02d}.jpg")
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64))
        paths.append(path)
    return paths


def make_mask_for_vggt(alpha_image, target=518):
    width, height = alpha_image.size
    max_dim = max(width, height)
    left = (max_dim - width) // 2
    top = (max_dim - height) // 2
    alpha = alpha_image.getchannel("A")
    canvas = Image.new("L", (max_dim, max_dim), 0)
    canvas.paste(alpha, (left, top))
    return canvas.resize((target, target), Image.Resampling.BILINEAR)


def prepare_views(paths, session, workdir):
    model_paths = []
    masks = []
    colors = []
    for index, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        if max(image.size) > 2400:
            scale = 2400 / max(image.size)
            image = image.resize(
                (int(image.width * scale), int(image.height * scale)),
                Image.Resampling.LANCZOS
            )
        cutout = remove(image, session=session).convert("RGBA")
        mask = make_mask_for_vggt(cutout)
        rgb = Image.new("RGBA", cutout.size, (255, 255, 255, 255))
        rgb.alpha_composite(cutout)
        out_path = str(Path(workdir) / f"view_{index:02d}.png")
        rgb.convert("RGB").save(out_path)
        model_paths.append(out_path)
        masks.append(np.asarray(mask) > 100)
        max_dim = max(rgb.size)
        left = (max_dim - rgb.width) // 2
        top = (max_dim - rgb.height) // 2
        color_canvas = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        color_canvas.paste(rgb.convert("RGB"), (left, top))
        colors.append(np.asarray(color_canvas.resize((518, 518), Image.Resampling.BILINEAR)))
    return model_paths, masks, colors


def build_colored_mesh(points, colors):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

    cloud = cloud.voxel_down_sample(voxel_size=0.004)
    if len(cloud.points) < 1500:
        raise gr.Error("Too few reliable foreground points. Try clearer photos with a plain background.")

    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.025, max_nn=40)
    )
    cloud.orient_normals_consistent_tangent_plane(30)

    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=8, scale=1.08, linear_fit=True
    )
    density = np.asarray(density)
    keep = density > np.quantile(density, 0.08)
    mesh.remove_vertices_by_mask(~keep)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    vertices = np.asarray(mesh.vertices)
    source_points = np.asarray(cloud.points)
    source_colors = np.asarray(cloud.colors)
    nearest = cKDTree(source_points).query(vertices, k=1)[1]
    vertex_colors = np.clip(source_colors[nearest] * 255.0, 0, 255).astype(np.uint8)

    result = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(mesh.triangles),
        vertex_colors=vertex_colors,
        process=True
    )
    result.remove_unreferenced_vertices()
    return result, len(source_points)


@spaces.GPU(duration=190)
def reconstruct(confidence_percentile):
    model, session = get_models()

    with tempfile.TemporaryDirectory(prefix="multiview_human_") as tempdir:
        paths = write_embedded_images(tempdir)
        model_paths, masks, colors = prepare_views(paths, session, tempdir)
        images = load_and_preprocess_images(model_paths, mode="pad").to(DEVICE)

        with torch.inference_mode():
            major = torch.cuda.get_device_capability()[0]
            dtype = torch.bfloat16 if major >= 8 else torch.float16
            with torch.autocast(device_type="cuda", dtype=dtype):
                predictions = model(images.unsqueeze(0))

        world_points = predictions["world_points"].float().squeeze(0).cpu().numpy()
        confidence = predictions["world_points_conf"].float().squeeze(0).cpu().numpy()

        all_points = []
        all_colors = []
        for i in range(len(paths)):
            points = world_points[i].reshape(-1, 3)
            conf = confidence[i].reshape(-1)
            mask = masks[i].reshape(-1)
            color = colors[i].reshape(-1, 3)

            finite = np.isfinite(points).all(axis=1) & np.isfinite(conf)
            if np.any(finite):
                threshold = np.percentile(conf[finite], float(confidence_percentile))
                keep = finite & mask & (conf >= threshold)
                all_points.append(points[keep])
                all_colors.append(color[keep])

        if not all_points:
            raise gr.Error("The model did not find enough consistent 3D points.")

        points = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0)

        if len(points) > 350000:
            chosen = np.random.default_rng(7).choice(len(points), 350000, replace=False)
            points, colors = points[chosen], colors[chosen]

        mesh, point_count = build_colored_mesh(points, colors)
        output_path = str(Path(tempdir) / "human_reconstruction.glb")
        mesh.export(output_path)
        persistent_path = str(Path(tempfile.gettempdir()) / f"human_reconstruction_{os.getpid()}.glb")
        Path(output_path).replace(persistent_path)

        del predictions, images, world_points, confidence
        gc.collect()
        torch.cuda.empty_cache()

    note = (
        f"Created a vertex-colored mesh from {len(paths)} embedded views. "
        f"Point samples used after downsampling: {point_count:,}. "
        "This is an experimental reconstruction; inspect and clean the mesh in Blender."
    )
    return persistent_path, persistent_path, note


with gr.Blocks(title="Multi-Photo Human 3D") as demo:
    gr.Markdown(
        "# Multi-Photo Human 3D\n"
        "Reconstructs a 3D mesh from the built-in set of photos and exports a GLB file for Blender."
    )
    confidence = gr.Slider(
        minimum=0, maximum=50, value=15, step=5,
        label="Confidence filtering percentile",
        info="Increase this to discard more uncertain points."
    )
    run = gr.Button("Reconstruct 3D mesh", variant="primary")
    preview = gr.Model3D(label="3D mesh preview", clear_color=[0.08, 0.08, 0.1, 1.0])
    download = gr.File(label="Download GLB for Blender")
    status = gr.Textbox(label="Result", interactive=False)
    run.click(reconstruct, inputs=[confidence], outputs=[preview, download, status])

demo.queue(max_size=4).launch()
