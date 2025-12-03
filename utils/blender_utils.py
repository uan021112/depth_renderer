import bpy
import numpy as np
from pathlib import Path
from mathutils import Matrix

from .io_utils import load_matrix_from_file
from .math_utils import calc_coord_transform


def cleanup_blender_memory():
    """Remove orphan Blender data blocks so multi-scene renders do not leak."""

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    if hasattr(bpy.data, "orphans_purge"):
        # Run twice to ensure nested dependencies (materials -> images, etc.) go away.
        for _ in range(2):
            bpy.data.orphans_purge(do_recursive=True)


def enable_gpu(device_type='CUDA', with_cpu=True):
    pref = bpy.context.preferences
    cpref = pref.addons["cycles"].preferences

    cpref.refresh_devices()

    activated = []
    for dev in cpref.devices:
        if dev.type == 'CPU':
            dev.use = with_cpu
        else:
            dev.use = True
            activated.append(dev.name)

    cpref.compute_device_type = device_type

    for scene in bpy.data.scenes:
        scene.cycles.device = 'GPU'

    print("Activated GPUs:", activated)
    return activated


def import_mesh(mesh_path):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    mesh_suffix = Path(mesh_path).suffix
    if mesh_suffix == '.obj':
        bpy.ops.wm.obj_import(filepath=mesh_path, forward_axis='Y', up_axis='Z')
    elif mesh_suffix == '.ply':
        bpy.ops.wm.ply_import(filepath=mesh_path, forward_axis='Y', up_axis='Z')
    else:
        raise KeyError("Undefined mesh suffix.")

    return bpy.context.object


def update_camera(cam_obj: bpy.types.Object, cam_matrix: Matrix, fx: float,
                  render_width: int, render_height: int, sensor_width: float):
    """Update camera pose/intrinsics without creating new Blender objects."""
    scene = bpy.context.scene
    cam_obj.matrix_world = cam_matrix
    scene.camera = cam_obj

    cam_data = cam_obj.data
    cam_data.lens = fx * sensor_width / render_width
    cam_data.sensor_width = sensor_width
    cam_data.sensor_height = sensor_width * render_height / render_width

    scene.render.resolution_x = render_width
    scene.render.resolution_y = render_height
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0

    return scene


def import_camera(intrinsic_path, pose_path, render_width=1920, render_height=1440,
                  sensor_width=36.0, cam_obj: bpy.types.Camera = None,
                  is_c2w_pose: bool=True):
    intrinsic = load_matrix_from_file(intrinsic_path)
    fx = intrinsic[0, 0]
    if is_c2w_pose:
        c2w = load_matrix_from_file(pose_path)
    else:
        c2w = np.linalg.inv(load_matrix_from_file(pose_path))
    
    cam_matrix = Matrix(c2w.tolist())

    if cam_obj is None:
        bpy.ops.object.camera_add()
        cam_obj = bpy.context.object

    scene = update_camera(cam_obj, cam_matrix, fx, render_width, render_height, sensor_width)
    return scene, cam_obj


def import_camera_scannet(intrinsic_path, pose_path, render_width=1920, render_height=1440,
                  sensor_width=36.0, cam_obj: bpy.types.Camera = None,
                  is_c2w_pose: bool=True):
    intrinsic = load_matrix_from_file(intrinsic_path)
    fx = intrinsic[0, 0]
    if is_c2w_pose:
        c2w = load_matrix_from_file(pose_path)
    else:
        c2w = np.linalg.inv(load_matrix_from_file(pose_path))

    magic_xform = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    c2w = c2w @ magic_xform
    
    cam_matrix = Matrix(c2w.tolist())

    if cam_obj is None:
        bpy.ops.object.camera_add()
        cam_obj = bpy.context.object

    scene = update_camera(cam_obj, cam_matrix, fx, render_width, render_height, sensor_width)
    return scene, cam_obj


def render_depth(scene: bpy.types.Scene, zNear: float, zFar: float, output_path: str,
                 idx: int = 0, enable_png: bool = False):
    if zFar <= zNear:
        raise ValueError(f"zFar ({zFar}) must be greater than zNear ({zNear})")

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale outputs so the file output node can overwrite cleanly
    stale_patterns = ["depth*.exr"]
    if enable_png:
        stale_patterns.append("depth*.png")
    for pattern in stale_patterns:
        for stale in output_dir.glob(pattern):
            stale.unlink()

    view_layer = scene.view_layers[0]
    view_layer.use_pass_z = True
    view_layer.use_pass_combined = False

    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    render_layer = tree.nodes.new(type="CompositorNodeRLayers")

    map_node = tree.nodes.new(type="CompositorNodeMapRange")
    map_node.use_clamp = True
    map_node.inputs["From Min"].default_value = zNear
    map_node.inputs["From Max"].default_value = zFar
    map_node.inputs["To Min"].default_value = 0.0
    map_node.inputs["To Max"].default_value = 1.0

    file_exr = tree.nodes.new(type="CompositorNodeOutputFile")
    file_exr.label = "DepthEXR"
    file_exr.base_path = str(output_dir)
    file_exr.file_slots[0].path = "depth"
    file_exr.format.file_format = 'OPEN_EXR'
    file_exr.format.color_mode = 'RGB'
    file_exr.format.color_depth = '32'

    if enable_png:
        file_png = tree.nodes.new(type="CompositorNodeOutputFile")
        file_png.label = "DepthPNG"
        file_png.base_path = str(output_dir)
        file_png.file_slots[0].path = "depth"
        file_png.format.file_format = "PNG"
        file_png.format.color_mode = "RGB"
        file_png.format.color_depth = "16"

    tree.links.new(render_layer.outputs["Depth"], file_exr.inputs["Image"])
    tree.links.new(render_layer.outputs["Depth"], map_node.inputs["Value"])
    if enable_png:
        tree.links.new(map_node.outputs["Value"], file_png.inputs["Image"])

    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    scene.render.image_settings.file_format = 'OPEN_EXR'
    bpy.ops.render.render(write_still=True)

    def _collect_new_file(idx, extension):
        matches = sorted(output_dir.glob(f"depth*.{extension}"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
        if not matches:
            raise RuntimeError(f"Failed to render depth {extension}")
        latest = matches[0]
        target = output_dir / f"frame-{idx:06d}.depth.{extension}"
        latest.rename(target)

    if enable_png:
        _collect_new_file(idx, 'png')
    _collect_new_file(idx, 'exr')
