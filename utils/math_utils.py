import numpy as np

def calc_coord_transform(A='RDF', B='RDF'):
    direction_map = {
        'R': [1, 0, 0],
        'L': [-1, 0, 0],
        'U': [0, 1, 0],
        'D': [0, -1, 0],
        'F': [0, 0, 1],
        'B': [0, 0, -1]
    }

    dataset_matrix = np.array([
        direction_map[A[0]],  
        direction_map[A[1]],  
        direction_map[A[2]] 
    ]).T
    
    target_matrix = np.array([
        direction_map[B[0]],
        direction_map[B[1]],
        direction_map[B[2]]
    ]).T
    
    def is_orthogonal(matrix):
        return np.allclose(matrix @ matrix.T, np.eye(3))
    
    if not is_orthogonal(dataset_matrix):
        raise ValueError("Dataset A system is not orthogonal")
    if not is_orthogonal(target_matrix):
        raise ValueError("Target A system is not orthogonal")
    
    M_A2B = target_matrix.T @ dataset_matrix
    M4x4_A2B = np.eye(4)
    M4x4_A2B[:3, :3] = M_A2B
    
    return M4x4_A2B