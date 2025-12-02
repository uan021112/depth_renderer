from ._3dscanner import parse_workspace_path_3dscanner
from .scannet import parse_workspace_path_scannet

parse_workspace_path_callback = {
    '3dscanner': parse_workspace_path_3dscanner,
    'scannet': parse_workspace_path_scannet,
}