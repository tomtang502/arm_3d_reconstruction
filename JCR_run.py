import os, torch
import utils.geometric_util as geomu
from utils.graph_util import plotty_graph_multistruct
from utils.dust3r_api import load_pose_from_exp_name
from utils.fix_scale_calib import *
from utils.scale_calib import *

import numpy as np
from configs.experiments_data_config import ArmDustrExpData

exp_config = ArmDustrExpData()
writing_file = 'output/dust3r_calib_loss.txt'

def jcr_run(exp_name, num_imgs, save_dir):
    
    ### Load arm end-effectors, camera poses, and point cloud (the last two are generated by dust3r)
    pose_data = exp_config.get_obs_config(exp_name)

    eef_poses_all = pose_data.poses + pose_data.additional_colmap_pose
    eef_poses_tor=geomu.pose_to_transform(torch.tensor(eef_poses_all))

    im_poses_tor_o, ptc_tor_o, rgb_colors, loc_info = load_pose_from_exp_name(exp_name, num_imgs)
    eef_sc_used, dust3r_sc_used, eef_nontest, eef_nontest_idx = scale_calib_pose_process(eef_poses_tor, 
                                                                                        im_poses_tor_o, 
                                                                                        pose_data.test_pt, 
                                                                                        pose_data.linearidx)

    assert eef_nontest.shape == im_poses_tor_o.shape, "Number of eef != Number of cam poses!"

    ### Solving for scale and then do caliberation
    T, scale, J, R_L, t_L = compute_arm(eef_sc_used, dust3r_sc_used)
    im_poses_tor_o[:,:3,3]=im_poses_tor_o[:,:3,3]*scale
    ptc_tor_o = ptc_tor_o*scale

    loss_info = f'{exp_name}_{num_imgs} trans loss: {t_L.mean()}, rot loss: {R_L.mean()}\n'
    print(loss_info)
    with open(writing_file, 'a') as file:
        file.write(loss_info) 


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
        'colors': rgb_colors,
        'pt_loc' : loc_info,
        'eef_poses': eef_nontest,
        'T' : T,
        'eef_idx': eef_nontest_idx,
        'J' : torch.tensor(J),
        'trans_L' : torch.tensor(t_L),
        'rot_L' : torch.tensor(R_L)
    }

    # Saving the dictionary of tensors to a file
    saving_loc = os.path.join(save_dir, f'{exp_name}_{num_imgs}.pth')
    torch.save(tensors_to_save, saving_loc)
    print("="*10)
    print(f"dust3r out saved at {saving_loc}")
    print("="*10)

if __name__ == "__main__":
    """
    Other experiments
    '8obj_divangs', '8obj_4cluster',
    '7obj_divangs', '7obj_4cluster',
    'shelf_divangs', 'shelf_4cluster'
    
    """
    out_dir = exp_config.dustr_out_pth
    if not os.path.exists(out_dir):
        print(f"making {out_dir}")
        os.makedirs(out_dir)
    exp_name_list = ['4obj_measure']
    for exp_name in exp_name_list:
        for i in range(8, 11, 2):    
            saving_loc = os.path.join(out_dir, f'{exp_name}_{i}.pth')
            print("Working on", saving_loc)
            if os.path.isfile(saving_loc):
                print(saving_loc, "already processed")
            else:
                jcr_run(exp_name=exp_name, num_imgs=i, save_dir=out_dir)
                #load_pose_from_exp_name(exp_name, 20)

