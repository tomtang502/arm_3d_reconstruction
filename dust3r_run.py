import sys, os, pickle, torch, argparse
import utils.chessboard as chessboard

import utils.geometric_util as geomu
from utils.graph_util import plotty_graph_multistruct, graph_double_struct
from utils.dust3r_api import load_pose_from_exp_name
from utils.fix_scale_calib import *
from utils.scale_calib import *

import numpy as np
from configs.experiments_data_config import ArmDustrExpData


exp_config = ArmDustrExpData()
# # Create the parser
# parser = argparse.ArgumentParser(description='Example script that accepts a string argument.')

# # Add an argument
# parser.add_argument('exp_name', type=str, help='An experiment name')
# parser.add_argument('num_imgs', type=int, help='Number of images to consider')

# # Execute the parse_args() method
# args = parser.parse_args()

# Store the argument in a variable
#exp_name = args.exp_name
exp_name = '8obj_divangs'
#num_imgs = args.num_imgs
num_imgs = 10

def dust3r_run(exp_name, num_imgs):
    
    ### Load arm end-effectors, camera poses, and point cloud (the last two are generated by dust3r)
    pose_data = exp_config.get_obs_config(exp_name)

    eef_poses_all = pose_data.poses + pose_data.additional_colmap_pose
    eef_poses_tor=geomu.pose_to_transform(torch.tensor(eef_poses_all))

    im_poses_tor_o, ptc_tor_o = load_pose_from_exp_name(exp_name, num_imgs)
    eef_sc_used, dust3r_sc_used, eef_nontest, eef_nontest_idx = scale_calib_pose_process(eef_poses_tor, 
                                                                                        im_poses_tor_o, 
                                                                                        pose_data.test_pt, 
                                                                                        pose_data.linearidx)

    # xyz = np.stack(geomu.tmatw2c_to_xyz(im_poses_tor_o))
    # xyz_eef = np.stack(geomu.tmatw2c_to_xyz(eef_nontest))
    #graph_double_struct(xyz, xyz_eef)
    print(f"{eef_nontest.shape} should be as same as {im_poses_tor_o.shape}")
    assert eef_nontest.shape == im_poses_tor_o.shape, "Number of eef != Number of cam poses!"


    ### Solving for scale and then do caliberation
    T, scale = computer_arm(eef_sc_used, dust3r_sc_used)
    im_poses_tor_o[:,:3,3]=im_poses_tor_o[:,:3,3]*scale
    ptc_tor_o = ptc_tor_o*scale

    dust3r_pose, dust3r_ptc = transpose_poses_ptc(im_poses_tor_o, ptc_tor_o, T)


    #Visualize constructed ptc
    pts_tor_n = dust3r_ptc[::300]
    cam_pos_n=dust3r_pose[:,:3,3]
    eff_poses_n=eef_nontest[:,:3,3]
    plotty_graph_multistruct([eff_poses_n, cam_pos_n, pts_tor_n], 
                            ["arm end-effector", "camera pose", "point cloud"],
                            [2, 2, 0.3])

    tensors_to_save = {
        'poses': dust3r_pose,
        'dense_pt': dust3r_ptc,
        'eef_poses': eef_nontest,
        'T' : T,
        'eef_idx': eef_nontest_idx
    }

    # Saving the dictionary of tensors to a file
    saving_loc = os.path.join("output/dust3r_saved_output", f'{exp_name}_{num_imgs}.pth')
    torch.save(tensors_to_save, saving_loc)
    print("="*10)
    print(f"dust3r out saved at {saving_loc}")
    print("="*10)

if __name__ == "__main__":
    """
    Done
    '8obj_divangs', '8obj_4cluster',
    '7obj_divangs', '7obj_4cluster',
    'shelf_divangs', 'shelf_4cluster',
    """
    exp_name_list = []
    for exp_name in exp_name_list:
        for i in range(10, 20):    
            ptc_pth = exp_config.get_ptc_output_path(exp_name)
            poses_pth = exp_config.get_cam_pose_path(exp_name)
            print(ptc_pth, poses_pth)
            if os.path.isfile(ptc_pth) and os.path.isfile(poses_pth):
                os.remove(ptc_pth)
                os.remove(poses_pth)
            dust3r_run(exp_name=exp_name, num_imgs=i)

