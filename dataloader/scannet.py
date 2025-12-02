from pathlib import Path

def parse_workspace_path_scannet(original_path, processed_path, scene_prefix: str='scene0000_00'):
    original = Path(original_path).expanduser() / scene_prefix
    processed = Path(processed_path).expanduser() / scene_prefix
    if not original.is_dir():
        raise FileNotFoundError(f"Workspace does not exist: {original}")
    if not processed.is_dir():
        raise FileNotFoundError(f"Workspace does not exist: {processed}")

    intrinsic_dir = processed / "intrinsic"
    pose_dir = processed / "pose"
    if not intrinsic_dir.is_dir():
        raise FileNotFoundError(f"Missing intrinsic directory: {intrinsic_dir}")
    if not pose_dir.is_dir():
        raise FileNotFoundError(f"Missing pose directory: {pose_dir}")

    mesh_path = original / f'{scene_prefix}_vh_clean.ply'
    if mesh_path is None:
        raise FileNotFoundError(
            f"Expected mesh file under {original}"
        )

    depth_path = processed / "depth"

    return str(mesh_path), str(intrinsic_dir), str(pose_dir), str(depth_path)

if __name__ == '__main__':
    a,b,c,d = parse_workspace_path_scannet('/home/tcluan/D-Data/ScanNet/scans', '/home/tcluan/D-Data/ExpLog/ScanNet_Processed', 'scene0000_00')
    print(a,b,c,d,sep='\n')