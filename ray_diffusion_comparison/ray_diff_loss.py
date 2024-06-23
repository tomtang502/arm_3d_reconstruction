import os, sys, torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import utils.geometric_util as geomu
from utils.graph_util import plotty_graph_multistruct
from utils.dust3r_api import load_pose_from_exp_name
from utils.fix_scale_calib import *
from utils.scale_calib import *

import numpy as np
from configs.experiments_data_config import ArmDustrExpData

rdiff_res = torch.load("rdiff_output.pt")

writing_file = '../output/raydiff_calib_loss.txt'
exp_names = [
        '8obj_divangs',
        '7obj_divangs',
        'shelf_divangs'
]
num_img_options = [8, 10, 12, 15]
exp_config = ArmDustrExpData()

for exp_name in exp_names:
    for num_imgs in num_img_options:
        pose_data = exp_config.get_obs_config(exp_name)

        eef_poses_all = pose_data.poses + pose_data.additional_colmap_pose
        eef_poses_tor=geomu.pose_to_transform(torch.tensor(eef_poses_all))

        im_poses_tor_o = rdiff_res[f"{exp_name}_{num_imgs}"]
        eef_sc_used, dust3r_sc_used, eef_nontest, eef_nontest_idx = scale_calib_pose_process(eef_poses_tor, 
                                                                                            im_poses_tor_o, 
                                                                                            pose_data.test_pt, 
                                                                                            pose_data.linearidx)

        assert eef_nontest.shape == im_poses_tor_o.shape, "Number of eef != Number of cam poses!"

        ### Solving for scale and then do caliberation
        T, scale, J, R_L, t_L = compute_arm(eef_sc_used, dust3r_sc_used)
        im_poses_tor_o[:,:3,3]=im_poses_tor_o[:,:3,3]*scale

        loss_info = f'{exp_name}_{num_imgs} trans loss: {t_L.mean()}, rot loss: {R_L.mean()}\n'
        print(loss_info)
        with open(writing_file, 'a') as file:
            file.write(loss_info) 