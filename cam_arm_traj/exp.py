import numpy as np
from move_arm import *

a = np.array([[1,2,3, 0, 0, 0], 
              [-1,-2,-1, 4, 5, 6]])
default_ori = np.array([0.0, 0.0, 0.0])


def convert_tag_to_arm_coordinates(traj, init_pos, init_ori = default_ori):
    # init_pos is based on arm coordinates
    traj = traj[:, [5, 4, 3, 1, 0, 2]]
    traj[:, 4] = -1.0 * traj[:, 4]
 
    shift = np.concatenate((default_ori[::-1], init_pos))
    shift = np.tile(shift, (traj.shape[0], 1))

    return traj + shift

test_init_pos = np.array([0.11, 0.0, 0.01])
#print(convert_tag_to_arm_coordinates(a, test_init_pos))

run_traj(np.array([0,0,0,0,0,0.1,0]), None, 10, True)