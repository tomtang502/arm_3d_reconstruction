import torch
import numpy as np
from dt_apriltags import Detector
import cv2
import os
import copy
from utils.geometric_util import *
from utils.graph_util import *
from utils.fix_scale_calib import *
from configs.experiments_data_config import ArmDustrExpData
from scipy.spatial.transform import Rotation as R
from utils.scale_calib import *

exp_config = ArmDustrExpData()

def tmat_loss(matrix1, matrix2):
    """
    Calculate the scalar loss between two transformation matrices.

    Args:
    matrix1, matrix2: Numpy arrays representing the 4x4 transformation matrices.

    Returns:
    Scalar loss/error between the matrices.
    """
    # Calculate translation difference
    translation1 = matrix1[:3, 3]
    translation2 = matrix2[:3, 3]
    translation_difference = np.linalg.norm(translation1 - translation2)

    # Calculate rotation difference
    rotation1 = matrix1[:3, :3]
    rotation2 = matrix2[:3, :3]
    rotation_difference = (rotation2@rotation1.T)
    # print(rotation_difference)
    # print(torch.arccos((torch.trace(rotation_difference) - 1.0) / 2.0))
    theta_rad = torch.acos((torch.trace(rotation_difference) - 1.0) / 2.0)
    #theta_deg = torch.rad2deg(theta_rad)
    
    return translation_difference, theta_rad

def run_comparison(exp_name, img_num):
    print('-'*90)
    exp_ref_name = 'apriltag' + '_' + exp_name.split('_')[1]
    if 'shelf' in exp_name:
        exp_ref_name = exp_ref_name + 'a'

    print(exp_name, exp_ref_name)

    print("Run Comparison")
    colmap_pose_dir = f"output/colmap_saved_output/{exp_name}_{img_num}.pth"
    dust3r_pose_dir = f"output/dust3r_saved_output/{exp_name}_{img_num}.pth"
    april_dir = f"output/apriltag_saved_output/{exp_ref_name}.pth"

    colmap_failed = False
    if os.path.isfile(colmap_pose_dir):
        colmap_out = torch.load(colmap_pose_dir)
        colmap_poses = colmap_out['poses']
        if colmap_poses.shape[0] < 4:
            print(exp_name, img_num, "colmap failed on short of images")
            colmap_failed = True

    else:
        print(exp_name, img_num, "colmap failed")
        colmap_failed = True

    dust3r_out = torch.load(dust3r_pose_dir)
    april_out = torch.load(april_dir)

    
    
    #dust3r_poses = dust3r_out['poses']
    dust3r_T = dust3r_out['T']
    ref_T = april_out['T']
    print("dust3r", dust3r_T)
    print("apriltag", ref_T)
    #print(colmap_T.shape, dust3r_T.shape, ref_T.shape)
    if not colmap_failed:
        colmap_T = colmap_out['T']
        print(f"{exp_name} with {img_num} images")
        c_t_diff, c_theta_rad_diff = tmat_loss(colmap_T, ref_T)
        print("colmap", colmap_T)
        print(f"colmap translational loss {c_t_diff}, rotational loss {c_theta_rad_diff}")
    d_t_diff, d_theta_rad_diff = tmat_loss(dust3r_T, ref_T)
    print(f"dust3r trans loss {d_t_diff}, rot loss {d_theta_rad_diff}")
    


if __name__ == "__main__":
    ############################################
    exp_name = "7obj_4cluster"
    img_num = 10
    ############################################
    exp_name_list = ['8obj_divangs',
    '8obj_4cluster',  

    '7obj_divangs',
    '7obj_4cluster',  
    
    'shelf_divangs',
    'shelf_4cluster']
    exp_name = exp_name_list[5]
    for img_num in range(10, 21):
        run_comparison(exp_name, img_num)
        print('-'*90)
    