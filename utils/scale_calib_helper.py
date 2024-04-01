import torch
import numpy as np
from scipy.linalg import inv
from numpy import dot, eye, zeros, outer
from numpy.linalg import inv

def log(R):
    # Rotation matrix logarithm
    theta = np.arccos((R[0,0] + R[1,1] + R[2,2] - 1.0)/2.0)
    return np.array([R[2,1] - R[1,2], 
                     R[0,2] - R[2,0], 
                     R[1,0] - R[0,1]]) * theta / (2*np.sin(theta))

def invsqrt(mat):
    u,s,v = np.linalg.svd(mat)
    return u.dot(np.diag(1.0/np.sqrt(s))).dot(v)

def calibrate(A, B):
    #transform pairs A_i, B_i
    N = len(A)
    M = np.zeros((3,3))
    for i in range(N):
        Ra, Rb = A[i][0:3, 0:3], B[i][0:3, 0:3]
        M += outer(log(Rb), log(Ra))

    Rx = dot(invsqrt(dot(M.T, M)), M.T)

    C = zeros((3*N, 3))
    d = zeros((3*N, 1))
    for i in range(N):
        Ra,ta = A[i][0:3, 0:3], A[i][0:3, 3]
        Rb,tb = B[i][0:3, 0:3], B[i][0:3, 3]
        C[3*i:3*i+3, :] = eye(3) - Ra
        d[3*i:3*i+3, 0] = ta - dot(Rx, tb)

    tx = dot(inv(dot(C.T, C)), dot(C.T, d))
    return Rx, tx.flatten()

def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Convert euler angles (roll, pitch, yaw) to a rotation matrix.
    """
    R_x = torch.tensor([[1, 0, 0],
                        [0, torch.cos(roll), -torch.sin(roll)],
                        [0, torch.sin(roll), torch.cos(roll)]])
    
    R_y = torch.tensor([[torch.cos(pitch), 0, torch.sin(pitch)],
                        [0, 1, 0],
                        [-torch.sin(pitch), 0, torch.cos(pitch)]])
    
    R_z = torch.tensor([[torch.cos(yaw), -torch.sin(yaw), 0],
                        [torch.sin(yaw), torch.cos(yaw), 0],
                        [0, 0, 1]])
    
    R = torch.matmul(R_z, torch.matmul(R_y, R_x))
    return R
    
def pose_to_transform(batch):
    """
    Construct a batch of transformation matrices from a batch of roll, pitch, yaw, x, y, z.
    The input batch should have the shape (n, 6), where each row is (roll, pitch, yaw, x, y, z).
    The output will have the shape (n, 4, 4), representing the transformation matrices.
    """
    n = batch.size(0)
    transformation_matrices = torch.zeros((n, 4, 4))
    
    for i in range(n):
        roll, pitch, yaw, x, y, z = batch[i]
        rotation_matrix = euler_to_rotation_matrix(roll, pitch, yaw)
        transformation_matrix = torch.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = torch.tensor([x, y, z])
        transformation_matrices[i] = transformation_matrix
    
    return transformation_matrices



def compute_A_B_Rx(eff_poses_tor,im_poses):
    A, B = [], []
    for i in range(1,len(im_poses)):
        p = eff_poses_tor[i-1], im_poses[i-1]
        n = eff_poses_tor[i], im_poses[i]
        A.append(dot(inv(p[0]), n[0]))
        B.append(dot(inv(p[1]), n[1]))
    
    N = len(A)
    M = np.zeros((3,3))
    for i in range(N):
        Ra, Rb = A[i][0:3, 0:3], B[i][0:3, 0:3]
        M += outer(log(Rb), log(Ra))
    
    Rx = dot(invsqrt(dot(M.T, M)), M.T)
    return(A, B, Rx, N)

def compute_cost(scale, A, B, Rx, N):
    C = zeros((3*N, 3))
    d = zeros((3*N, 1))
    for i in range(N):
        Ra,ta = A[i][0:3, 0:3], A[i][0:3, 3]
        Rb,tb = B[i][0:3, 0:3], B[i][0:3, 3]
        C[3*i:3*i+3, :] = eye(3) - Ra
        d[3*i:3*i+3, 0] = ta - dot(Rx, scale*tb)

    tx = dot(inv(dot(C.T, C)), dot(C.T, d))
    J = np.mean(np.linalg.norm(np.dot(C, tx) - d)**2)
    return(J, tx)

def get_dust3r_useidx(num_eef, num_dust3r, test_idx, linear_idx):
    eef_useful_idx, dust3r_useful_idx, eef_nontest_idx = [], [], []
    cur_idx = 0
    for i in range(num_eef):
        if i not in test_idx and cur_idx < num_dust3r:
            eef_nontest_idx.append(i)
            if i not in linear_idx:
                    dust3r_useful_idx.append(cur_idx)
                    eef_useful_idx.append(i)
            cur_idx += 1
            
    return eef_useful_idx, dust3r_useful_idx, eef_nontest_idx

def scale_calib_pose_process(eef_poses_tr, dust3r_poses, test_idx, linear_idx):
    eef_useful_idx, dust3r_useful_idx, eef_nontest_idx = get_dust3r_useidx(len(eef_poses_tr), 
                                                                           len(dust3r_poses), 
                                                                           test_idx, linear_idx)
    eef_sc_used = eef_poses_tr[eef_useful_idx]
    dust3r_sc_used = dust3r_poses[dust3r_useful_idx]
    eef_nontest = eef_poses_tr[eef_nontest_idx]
    return eef_sc_used, dust3r_sc_used, eef_nontest, eef_nontest_idx


def get_colmap_useidx(num_eef, col_selected_idx, test_idx, linear_idx):
    eef_useful_idx, colmap_useful_idx, eef_nontest_idx = [], [], []
    cur_idx = 0
    for i in range(num_eef):
        if i not in test_idx and i in col_selected_idx:
            eef_nontest_idx.append(i)
            if i not in linear_idx:
                if cur_idx < len(col_selected_idx):
                    colmap_useful_idx.append(cur_idx)
                    eef_useful_idx.append(i)
            cur_idx += 1
            
    return eef_useful_idx, colmap_useful_idx, eef_nontest_idx

def scale_calib_pose_process_col(eef_poses_tr, colmap_poses, col_selected_idx, test_idx, linear_idx):
    linear_idx = linear_idx + [25, 26]
    eef_useful_idx, colmap_useful_idx, eef_nontest_idx = get_colmap_useidx(len(eef_poses_tr), 
                                                                           col_selected_idx, 
                                                                           test_idx, linear_idx)
    eef_sc_used = eef_poses_tr[eef_useful_idx]
    colmap_sc_used = colmap_poses[colmap_useful_idx]
    eef_nontest = eef_poses_tr[eef_nontest_idx]
    return eef_sc_used, colmap_sc_used, eef_nontest