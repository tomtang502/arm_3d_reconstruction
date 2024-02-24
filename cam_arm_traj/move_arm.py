import sys
import torch
sys.path.append("../z1_sdk/lib")
import unitree_arm_interface
import time
import numpy as np


print("Relative path checked")
"""
z 104.5, 575.3
arm._ctrlComp.dt = 0.002
cartesian space init posture:  0.00000 -0.07643  0.00001  0.08638  0.00000  0.17800
"""
default_ori = np.array([0,0,0])

def moveJ_in_dt(arm, tloc, dt = 1000):
    # arm.labelRun("forward")
    gripper_pos = 1.0
    jnt_speed = 2.0
    #arm.MoveJ(np.array([0.5,0.1,0.1,0.05,0,0.67]), gripper_pos, jnt_speed)
    arm.MoveJ(tloc, gripper_pos, jnt_speed)
    #0.0, 0.0, 0.0, 0.2, -0.2, 0.15
    # gripper_pos = -1.0
    # cartesian_speed = 0.5
    # arm.MoveL(np.array([0,0,0,0.45,-0.2,0.2]), gripper_pos, cartesian_speed)
    # gripper_pos = 0.0
    # arm.MoveC(np.array([0,0,0,0.45,0,0.4]), np.array([0,0,0,0.45,0.2,0.2]), gripper_pos, cartesian_speed)

def convert_tag_to_arm_coordinates(traj, init_pos, init_ori = default_ori):
    # init_pos is based on arm coordinates
    traj = traj[:, [5, 4, 3, 1, 0, 2]]
    traj[:, 4] = -1.0 * traj[:, 4]
    zeros_column = np.zeros((traj.shape[0], 1))

    shift = np.concatenate((default_ori[::-1], init_pos))
    shift = np.tile(shift, (traj.shape[0], 1))

    traj = traj #+ shift

    return np.hstack((traj, zeros_column))


def run_traj(all_traj, init_loc=None, steps_lim = 1000000):
    arm_traj = convert_tag_to_arm_coordinates(all_traj, init_loc)
    np.set_printoptions(precision=3, suppress=True)
    arm =  unitree_arm_interface.ArmInterface(hasGripper=True)
    armState = unitree_arm_interface.ArmFSMState
    print(all_traj.shape[0])
    arm.loopOn()
    arm.backToStart()
    arm.startTrack(armState.CARTESIAN)
    angular_vel = 0.3
    linear_vel = 0.3
    for i in range(0, min(all_traj.shape[0], steps_lim)):
        #print(arm_traj[i])
        arm.cartesianCtrlCmd(arm_traj[i], angular_vel, linear_vel)
        time.sleep(arm._ctrlComp.dt)
    # for i in range(0, min(all_traj.shape[0], steps_lim)):
    #     print(i)
    #     moveJ_in_dt(arm, arm_traj[i])
    #     time.sleep(arm._ctrlComp.dt)
    arm.backToStart()
    arm.loopOff()

"""
(roll pitch yaw x y z) + (gripper if cartisian)
"""
if __name__ == "__main__":
    test_init_pos = [0.21, 0.0, 0.1]

    sample_name = "square0"
    all_traj, inter_points_list, orientation_values = torch.load(f'sample_motion/{sample_name}.pt')
    div = 1 #all_traj.shape[0]//
    all_traj = all_traj[::div, :]
    print(all_traj.shape)
    # arm =  unitree_arm_interface.ArmInterface(hasGripper=True)
    # armState = unitree_arm_interface.ArmFSMState
    # arm.loopOn()
    # arm.backToStart()
    # arm.backToStart()
    # arm.loopOff()

    run_traj(all_traj, test_init_pos)
    