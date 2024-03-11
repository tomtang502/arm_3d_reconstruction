import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

def flip_axis(L):
    print(L[:, 2].shape)
    L = np.hstack((L[:, 0].reshape(-1, 1), L[:, 1].reshape(-1, 1), L[:, 2].reshape(-1, 1)))
    return L

# imput as a list of [row, pitch, yaw]
def rpy_to_rot_matrix(ori):
    r = R.from_euler('XYZ', ori, degrees=False)
    return torch.tensor(r.as_matrix())
    
def pose_to_transform(pose_batch):
    # Unpack the pose components
    pos = pose_batch[:, 3:6]

    # Convert RPY to rotation matrices
    rotation_matrices = rpy_to_rot_matrix(pose_batch[:, :3])

    # Create the transformation matrices
    transform_matrices = torch.zeros((pose_batch.shape[0], 4, 4), dtype=torch.float)
    transform_matrices[:, :3, :3] = rotation_matrices
    transform_matrices[:, :3, 3] = pos
    transform_matrices[:, 3, 3] = 1.0
    return transform_matrices

def rotation_matrix_difference(R1, R2):
    # Compute the difference between the two rotation matrices
    diff = np.dot(R1.T, R2) - np.eye(3)

    # Calculate the Frobenius norm of the difference
    frobenius_norm = np.linalg.norm(diff, 'fro')

    return frobenius_norm

def euler_angle_error(euler_angles1, euler_angles2):
    # Ensure both inputs are numpy arrays
    euler_angles1 = np.array(euler_angles1)
    euler_angles2 = np.array(euler_angles2)

    # Compute the absolute difference for each angle component
    absolute_difference = np.abs(euler_angles1 - euler_angles2)

    # Calculate the Euclidean norm of the absolute difference
    error_norm = np.linalg.norm(absolute_difference)

    return error_norm


def residual_error(T_R, A, B, Z):
    """
    Compute the residual error for the rotation matrix R.
    
    Parameters:
    R (base to world): torch.Tensor
        The estimated rotation matrix of shape (3, 3).
    A (world to cam), B (base to gripper): torch.Tensor
        3D rotation tensors of shape (N, 3, 3) representing sequences of 
        transformations A and B respectively.
    Z (gripper to cam): torch.Tensor 3D rotation tensors of shape (N, 3, 3) 
        representing sequences of rotation respectively.
    
    Returns:
    error : float
        The average residual error.
    """
    N = A.shape[0]
    errors = []
    for i in range(N):
        AR = np.matmul(A[i], T_R)
        ZB = np.matmul(Z, B[i])
        
        r1 = R.from_matrix(AR)
        r2 = R.from_matrix(ZB)

        # Convert the rotation to Euler angles (in radians)
        euler_angles_1 = r1.as_euler('xyz', degrees=False)
        euler_angles_2 = r2.as_euler('xyz', degrees=False)
        error = euler_angle_error(euler_angles_1, euler_angles_2)
        #print(error)
        errors.append(error)
    return sum(errors) / N

def apply_transform_pt(point, transformation_matrix):
    """
    Apply a transformation matrix to a point from a point cloud.
    
    Args:
    - point: numpy array representing the point coordinates [x, y, z]
    - transformation_matrix: 4x4 numpy array representing the transformation matrix
    
    Returns:
    - transformed_point: numpy array representing the transformed point coordinates [x', y', z']
    """
    # Append 1 to make it homogeneous coordinates
    point_homogeneous = np.append(point, 1)
    
    # Apply the transformation by matrix multiplication
    transformed_point_homogeneous = np.matmul(transformation_matrix, point_homogeneous)
    
    # Extract the transformed coordinates
    transformed_point = transformed_point_homogeneous[:3]
    
    return transformed_point

def transform_list_of_matrices_to_xyz(transformation_matrices):
    """
    Convert a list of transformation matrices to a list of XYZ coordinates.
    
    Args:
    - transformation_matrices: List of 4x4 numpy arrays representing the transformation matrices
    
    Returns:
    - coordinates_list: List of numpy arrays representing the XYZ coordinates
    """
    # Fixed point for transformation
    fixed_point = np.array([0, 0, 0])
    
    # List to store the XYZ coordinates
    coordinates_list = []
    
    # Apply each transformation matrix to the fixed point
    for transformation_matrix in transformation_matrices:
        transformed_point = apply_transform_pt(fixed_point, transformation_matrix)
        #transformed_point = transformation_matrix[:3, 3]
        coordinates_list.append(transformed_point)
    return coordinates_list
