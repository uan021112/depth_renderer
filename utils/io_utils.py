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

def get_intrinsic_file_path(intrinsic_path, idx):
    return Path(intrinsic_path) / f'frame-{idx:06d}.intrinsic_color.txt'


def get_pose_file_path(pose_path, idx):
    return Path(pose_path) / f'frame-{idx:06d}.pose.txt'