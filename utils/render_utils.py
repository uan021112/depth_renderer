import glob
import os
from pathlib import Path

import yaml

from dataloader import parse_workspace_path_callback
from .io_utils import get_intrinsic_file_path, get_pose_file_path
from .blender_utils import import_mesh, import_camera, import_camera_scannet, render_depth, cleanup_blender_memory


def render_single_view(cfg_path='/home/tcluan/C-Code/tools/pt_utils/configs/render_single_view.yaml'):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    import_mesh(cfg['scene']['mesh_path'])
    scene, _ = import_camera(
        cfg['scene']['intrinsic_path'],
        cfg['scene']['pose_path'],
        cfg['render']['width'],
        cfg['render']['height'],
        cfg['render']['sensor_width'],
    )
    render_depth(scene, cfg['render']['zNear'], cfg['render']['zFar'], cfg['output']['depth_path'])


def render_scene_3DScanner(cfg_path='/home/tcluan/C-Code/tools/pt_utils/configs/render_batch_3DScanner.yaml'):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    mesh_path, intrinsic_path, pose_path, depth_path = parse_workspace_path_callback['3dscanner'](cfg['scene']['workspace_path'])
    import_mesh(mesh_path)
    num_frame = len(glob.glob(pose_path + '/*.txt'))
    cam_obj = None
    for idx in range(num_frame):
        intrinsic_file = get_intrinsic_file_path(intrinsic_path, idx)
        pose_file = get_pose_file_path(pose_path, idx)
        scene, cam_obj = import_camera(
            intrinsic_file,
            pose_file,
            cfg['render']['width'],
            cfg['render']['height'],
            cfg['render']['sensor_width'],
            cam_obj=None if idx == 0 else cam_obj,
        )
        render_depth(
            scene,
            cfg['render']['zNear'],
            cfg['render']['zFar'],
            depth_path,
            idx,
            enable_png=True,
        )


def render_scene_Scannet(cfg, scene_prefix: str='scene0000_00'):
    mesh_path, intrinsic_path, pose_path, depth_path = parse_workspace_path_callback['scannet'](
        cfg['scene']['original_scannet_path'],
        cfg['scene']['processed_scannet_path'],
        scene_prefix
        )
    import_mesh(mesh_path)
    num_frame = len(glob.glob(pose_path + '/*.txt'))
    cam_obj = None
    intrinsic_file = Path(intrinsic_path) / 'intrinsic_color.txt'

    try:
        for idx in range(num_frame):
            pose_file = get_pose_file_path(pose_path, idx)
            scene, cam_obj = import_camera_scannet(
                intrinsic_file,
                pose_file,
                cfg['render']['width'],
                cfg['render']['height'],
                cfg['render']['sensor_width'],
                cam_obj=None if idx == 0 else cam_obj,
            )
            render_depth(
                scene,
                cfg['render']['zNear'],
                cfg['render']['zFar'],
                depth_path,
                idx,
                # enable_png=True,
                enable_png=True if idx == 0 else False,
            )
    finally:
        cleanup_blender_memory()


def render_dataset_Scannet(cfg_path='/home/tcluan/C-Code/tools/pt_utils/configs/render_batch_Scannet.yaml'):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    scene_prefix_list = glob.glob(os.path.join(cfg['scene']['processed_scannet_path'], '*'))
    scene_prefix_list.sort()
    for idx in range(cfg['scene']['start_from'], min(cfg['scene']['end_with'], len(scene_prefix_list))):
        scene_prefix = Path(scene_prefix_list[idx]).name
        render_scene_Scannet(cfg, scene_prefix)