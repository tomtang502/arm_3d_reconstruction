import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

def flip_axis(L):
    print(L[:, 2].shape)
    L = np.hstack((L[:, 0].reshape(-1, 1), L[:, 1].reshape(-1, 1), L[:, 2].reshape(-1, 1)))
    return L

"""
Convert euler angles (roll, pitch, yaw) to a rotation matrix.
[IN] ori: (row, pitch, yaw) in an array, tuple, or list
"""
def rpy_to_rot_matrix(ori):
    r = R.from_euler('xyz', ori, degrees=False)
    return torch.tensor(r.as_matrix())

"""
Construct a batch of transformation matrices from a batch of roll, pitch, yaw, x, y, z.
[IN] pose_batch: should have shape (n, 6), where each row is (roll, pitch, yaw, x, y, z).
[OUT] A batch with shape (n, 4, 4) representing the transformation matrices.
"""
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

"""
Compute the residual error for the rotation matrix R.
[IN] T_R (base to world): the estimated rotation matrix of shape (3, 3).
[IN] A (world to cam): 3D rotation matrix of shape (N, 3, 3) representing sequences of 
        rortational transformations.
[IN] B (base to gripper): D rotation matrix of shape (N, 3, 3) representing sequences of 
        rotational transformations.
[IN] Z (gripper to cam): 3D rotation matrix of shape (N, 3, 3) representing sequences of 
        rotation respectively.
[OUT] error as a float represent the average residual error.
"""
def residual_error(T_R, A, B, Z):
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

"""
Apply a transformation matrix to a point from a point cloud.
[IN] point: numpy array representing the point coordinates [x, y, z]
[IN] transformation_matrix: 4x4 numpy array representing the transformation matrix
[OUT] transformed_point: numpy array representing the transformed point coordinates [x', y', z']
"""
def apply_transform_pt(point, transformation_matrix):
    # Append 1 to make it homogeneous coordinates
    point_homogeneous = np.append(point, 1)
    
    # Apply the transformation by matrix multiplication
    transformed_point_homogeneous = np.matmul(transformation_matrix, point_homogeneous)
    
    # Extract the transformed coordinates
    transformed_point = transformed_point_homogeneous[:3]
    
    return transformed_point

"""
Convert a list of transformation matrices to a list of XYZ coordinates.
[IN] transformation_matrices: List of 4x4 numpy arrays representing the transformation matrices
[OUT] coordinates_list: List of numpy arrays representing the XYZ coordinates
"""
def tmatw2c_to_xyz(transformation_matrices):
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

"""
Compute the scale factor given the following
[IN] cam_poses is an array of world to camera transformation marix.
[IN] linear_idx_axis index into the w3c (cam_poses) to get corresponding transformation 
    matrices that have cam poses moving parallel to the specified axis.
[IN] axis_d is the distance between each consecutive two cam poses in real world.
    (here the variable name use x becuase it is the pose example we have in cam_pose config)
[OUT] Scale factor to scale the w2c transformation matrix.
"""
def get_scale_factor(cam_poses, linear_idx_x, x_d):

    sum_dist = 0
    n = len(linear_idx_x)
    for i in range(n - 1):
        idx1 = linear_idx_x[i]
        idx2 = linear_idx_x[i+1]
        pt2pt_trans = np.matmul(torch.linalg.pinv(cam_poses[idx1]), cam_poses[idx2])
        vec = apply_transform_pt([0.0, 0.0, 0.0], pt2pt_trans)
        #print(idx, cam_xyz_L[idx, cam_xyz_L[idx + 1,0])
        sum_dist += np.linalg.norm(vec)
    scale_factor = x_d/(sum_dist/float(n-1))
    return scale_factor

"""
Applies a transformation to a set of 3D points.
[IN] points: A torch tensor of size (n, 3) representing n 3D points.
[IN] matrix: A torch tensor of size (4, 4) representing the transformation matrix.
[OUT] A torch tensor of size (n, 3) of transformed 3D points.
"""
def transform_points(points, matrix):
    
    # Check if the inputs are torch tensors
    if not isinstance(points, torch.Tensor) or not isinstance(matrix, torch.Tensor):
        raise ValueError("Both points and matrix must be torch.Tensor objects.")
    
    # Check the shape of the points and the matrix
    if points.shape[1] != 3 or matrix.shape != (4, 4):
        raise ValueError("Invalid shape for points or matrix. Points should be (n, 3) and matrix should be (4, 4).")
    
    # Add an extra dimension of ones to the points tensor to make it compatible with the transformation matrix
    ones = torch.ones(points.shape[0], 1, dtype=points.dtype, device=points.device)
    points_homogeneous = torch.cat([points, ones], dim=1)  # Convert points to homogeneous coordinates
    
    # Apply the transformation matrix to the points
    transformed_points_homogeneous = torch.mm(points_homogeneous, matrix.t())  # Multiply by the transpose of the matrix
    
    # Convert back from homogeneous coordinates by dropping the last dimension
    transformed_points = transformed_points_homogeneous[:, :3]
    
    return transformed_points