from pathlib import Path

def parse_workspace_path_3dscanner(workspace_path: str):
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