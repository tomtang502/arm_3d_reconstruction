import numpy as np
from move_arm import *
import torch

a = np.array([[1,2,3, 0, 0, 0], 
              [-1,-2,-1, 4, 5, 6]])
default_ori = np.array([0.0, 0.0, 0.0])


# def convert_tag_to_arm_coordinates(traj, init_pos, init_ori = default_ori):
#     # init_pos is based on arm coordinates
#     traj = traj[:, [5, 4, 3, 1, 0, 2]]
#     traj[:, 4] = -1.0 * traj[:, 4]
 
#     shift = np.concatenate((default_ori[::-1], init_pos))
#     shift = np.tile(shift, (traj.shape[0], 1))

#     return traj + shift

# test_init_pos = np.array([0.11, 0.0, 0.01])
# #print(convert_tag_to_arm_coordinates(a, test_init_pos))

# run_traj(np.array([0,0,0,0,0,0.1,0]), None, 10, True)

# This does not consider orientation values
# Return a list of 3d coordinates key points on trajectory
def cal_vel(pt1, pt2, first_move=False, ori = np.array([0,0,0])):
    # from pt1 to pt2
    v = pt2
    if not first_move:
        v = v - pt1
    
    return np.hstack([v, ori])
    


def inter_points_format(inter_points_list):
    kpt = []
    for inter_points in inter_points_list:
        mean_inter_p=inter_points.mean(axis=0)
        mean_inter_p[0] = mean_inter_p[0] * -1.
        print(mean_inter_p)
        kpt.append(mean_inter_p[[1, 0, 2],])

    return kpt
all_traj, inter_points_list, orientation_values = torch.load(f'sample_motion/23triangle0.pt')
a = [np.array([1,2,3]), np.array([4,5,6])]
kps = inter_points_format(inter_points_list)
print(kps)
print(cal_vel(kps[0], kps[1], False, np.array([10, 10, 10])))