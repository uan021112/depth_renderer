import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from utils.blender_utils import enable_gpu
from utils.render_utils import render_scene_3DScanner

    
if __name__ == '__main__':
    enable_gpu()
    render_scene_3DScanner(cfg_path='/home/tcluan/C-Code/tools/pt_utils/configs/render_batch_3DScanner.yaml')