import numpy as np
from pathlib import Path

def load_matrix_from_file(file_path):
    matrix = []
    
    with open(file_path, 'r') as f:
        for line in f:
            row = line.strip().split()
            row = [float(x) for x in row]
            matrix.append(row)
    
    return np.array(matrix)

def parse_workspace_path(workspace_path: str):
    workspace = Path(workspace_path).expanduser()
    print(workspace)
    if not workspace.is_dir():
        raise FileNotFoundError(f"Workspace does not exist: {workspace}")

    intrinsic_dir = workspace / "intrinsic"
    pose_dir = workspace / "pose"
    if not intrinsic_dir.is_dir():
        raise FileNotFoundError(f"Missing intrinsic directory: {intrinsic_dir}")
    if not pose_dir.is_dir():
        raise FileNotFoundError(f"Missing pose directory: {pose_dir}")

    mesh_path = workspace / "textured_output.obj"
    if mesh_path is None:
        raise FileNotFoundError(
            f"Expected mesh file 'textured_output.obj' or 'mesh.obj' under {workspace}"
        )

    depth_path = workspace / "depth"

    return str(mesh_path), str(intrinsic_dir), str(pose_dir), str(depth_path)


def get_camera_param_file_path(intrinsic_path, pose_path, idx):
    intrinsic_file = Path(intrinsic_path) / f'frame-{idx:06d}.intrinsic_color.txt'
    pose_file = Path(pose_path) / f'frame-{idx:06d}.pose.txt'
    
    return intrinsic_file, pose_file