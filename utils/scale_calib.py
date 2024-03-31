import sys, os
import torch
import numpy as np
import plotly.graph_objects as go

from numpy import dot, eye
from numpy.linalg import inv

import sys, os
project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_folder)
from utils.scale_calib_helper import *

def computer_arm(eef_poses_selected, w2c_poses_selected, colmap=False):
    
    w2c_poses_selected_scale_cpy = w2c_poses_selected.clone()
    if colmap:
        vals_sc=np.linspace(0.001,0.5,20000).reshape(-1)
    else:
        vals_sc=np.linspace(1,4,20000).reshape(-1)
    A, B, Rx, N=compute_A_B_Rx(eef_poses_selected,w2c_poses_selected_scale_cpy)


    #we can just brute force
    costs_list=[]
    tx_list=[]
    for i in range(len(vals_sc)):
        J_res, tx_res=compute_cost(vals_sc[i],A, B, Rx, N)
        costs_list.append(J_res)
        tx_list.append(tx_res)
    costs_tor=torch.tensor(costs_list)

    min_inds=torch.argmin(costs_tor)
    scale=vals_sc[min_inds]
    #scale=0.025
    print("scale found: {}".format(scale))

    w2c_poses_selected[:, :3,3]=w2c_poses_selected[:, :3, 3] * scale
    A, B, Rx, N=compute_A_B_Rx(eef_poses_selected, w2c_poses_selected)
    J,tx=compute_cost(1.,A, B, Rx, N) # the scale factor is just 1 as we have already rescaled

    X = eye(4)
    X[0:3, 0:3] = Rx
    X[0:3, -1] = tx.reshape(-1)

    tmp_list=[]
    for i in range(len(w2c_poses_selected)):
        rob = eef_poses_selected[i]
        obj = w2c_poses_selected[i]
        tmp = dot(rob, dot(X, inv(obj)))
        tmp_list.append(tmp)
    tmp_tor=torch.tensor(np.array(tmp_list))
    world_pose=tmp_tor.mean(dim=0)

    return world_pose, scale

if __name__ == "__main__":
    """
    '8obj_divangs' : 'diverse_ori_sa',
    '8obj_4cluster': 'fourcluster_ori_sa', 
    '8obj_backonly': 'backonly_ori_sa', 

    '7obj_divangs' : 'diverse_ori_sa',
    '7obj_4cluster': 'fourcluster_ori_sa', 
    '7obj_backonly': 'backonly_ori_sa', 
    
    # center of april tag about 0.15 m from base center
    'apriltag_divangs' : 'diverse_ori_sa',
    'apriltag_4cluster': 'fourcluster_ori_sa',
    'apriltag_backonly': 'backonly_ori_sa',
    
    'shelf_divangs' : 'shelf_div_sa',
    'shelf_4cluster': 'shelf_4cl_sa',
    'shelf_backonly': 'backonly_ori_sa',

    """
    exp_name = '7obj_divangs'
    from configs.experiments_data_config import ArmDustrExpData

    exp_config = ArmDustrExpData()

    pose_data = exp_config.get_obs_config(exp_name)
    eef_poses = pose_data.poses

    eff_poses_tor_o=pose_to_transform(torch.tensor(eef_poses))
    dust3r_pose_dir = f"../output/dust3r_saved_output/{exp_name}.pth"

    dust3r_out = torch.load(dust3r_pose_dir)
    im_poses_o = dust3r_out['poses']  

    eef_sc_used, dust3r_sc_used, eef_nontest = scale_calib_pose_process(eff_poses_tor_o, im_poses_o, 
                                                                        pose_data.test_pt, 
                                                                        pose_data.linearidx)
  
    eef_poses_selected_tr=eef_sc_used
    w2c_poses_selected = dust3r_sc_used

    world_pose, scale = computer_arm(eef_poses_selected_tr, w2c_poses_selected)
    im_poses_o[:,:3,3]=im_poses_o[:,:3,3]*scale

    c_pos=world_pose.float()@(im_poses_o)
    c_pos_n=c_pos[:,:3,3]
    eff_poses_n=eef_nontest[:,:3,3]
    print(eef_nontest.shape, c_pos.shape)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(x=c_pos_n[:, 0].detach().cpu(),  # X coordinates
        y=c_pos_n[:, 1].detach().cpu(),  # Y coordinates
        z=c_pos_n[:, 2].detach().cpu(),  # Z coordinates
                            mode='markers', marker=dict(size=2, color='red'),name="cams"))
    fig.add_trace(go.Scatter3d(x=eff_poses_n[:, 0].detach().cpu(),  # X coordinates
        y=eff_poses_n[:, 1].detach().cpu(),  # Y coordinates
        z=eff_poses_n[:, 2].detach().cpu(),  # Z coordinates
                            mode='markers', marker=dict(size=2, color='blue'),name="eef"))

    fig.update_layout(scene=dict(
                        xaxis=dict(range=[-0.7, 0.7]),
                        yaxis=dict(range=[-0.7, 0.7]),
                        zaxis=dict(range=[0, 1.2])
    ))

    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                    title="Camera Positions and Orientations")

    fig.update_layout(width=1000, height=700)
    fig.show()