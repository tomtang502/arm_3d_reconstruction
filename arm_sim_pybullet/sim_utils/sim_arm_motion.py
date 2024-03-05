#################### Arm Motion Implementation for Simulation ####################

import os, sys, time
import torch
import numpy as np
proj_dir = os.path.dirname(os.path.realpath(__file__)) + "/../.."
sys.path.append(proj_dir + "/utils")
from arm_motion_planning import plan_motion, joint_to_pose, pose_shape
# /home/tomtang/Documents/droplab/z1_teaching/utils/arm_motion_planning.py
# /home/tomtang/Documents/droplab/z1_teaching/arm_sim_pybullet/sim_utils/sim_arm_motion.py
base_vel = torch.tensor([0.0]).double() # base is fixed
lower_hand_vel = torch.tensor([0.0]).double() # lower hand cannot be moved
"""
---------------------------------------------------
Those are headers of some functions from unitree_arm_interface
---------------------------------------------------
/*
* Function: compute end effector frame (used for current spatial position calculation)
* Inputs: q: current joint angles
*         index: it can set as 0,1,...,6
*           if index ==  6, then compute end efftor frame,
*           else compute joint_i frame
* Returns: Transfomation matrix representing the end-effector frame when the joints are
*				at the specified coordinates
*/
HomoMat forwardKinematics(Vec6 q, int index = 6);

/*
 * Function: Gives the space Jacobian
 * Inputs: q: current joint angles
 * Returns: 6x6 Spatial Jacobian
 */
Mat6 CalcJacobian(Vec6 q);
"""
"""
plan_motion(chain, init_state, init_pos, target_pos, dt=0.002, const_vel=0.10, 
                tmax=20, pos_diff_epsilon=0.0001, pose_shape=pose_shape, 
                joint_angle_shape=joint_angle_shape, 
                to_numpy=True, arm_Jacobian=None, forward_kin = None)

"""
def getQ(config, p, zid, jidxs):
    jangs_sim = torch.tensor([jinfo[0] for jinfo in p.getJointStates(zid, jidxs)]).double()
    iniq = initq = [config.initq[i] for i in jidxs]
    cur_state = torch.tensor(initq).double() + jangs_sim
    return cur_state

# move from one point to another point
def move_2p_grip(p_f, config, p, zid, jtm_idxs, grip_idx, arm_model, 
                 gripper_target=-1.0):
    init_state = getQ(config, p, zid, jtm_idxs)
    
    #from unitree as unitree format (row, pitch, yaw, x, y, z)
    p_i = joint_to_pose(init_state, arm_model.forwardKinematics)
    init_pos = torch.tensor(p_i.reshape((pose_shape[0],))).double()
    target_pos = torch.tensor(p_f).double()

    _, jangs_pos_list, _ = plan_motion(None, init_state, init_pos, target_pos, 
                                       dt=config.dt, tmax=config.tmax, const_vel=config.const_vel, 
                                       pos_diff_epsilon=0.0018, to_numpy=False, 
                                       arm_Jacobian=arm_model.CalcJacobian,
                                       forward_kin=arm_model.forwardKinematics)
    
    target_state = jangs_pos_list[-1]
    duration = len(jangs_pos_list)
    to_jangs_grip(target_state, config, p, zid, jtm_idxs, grip_idx, arm_model,
                  gripper_target=gripper_target, init_state=init_state, duration=duration)

def to_jangs_grip(jangs, config, p, zid, jtm_idxs, grip_idx, arm_model, 
                  gripper_target=-1.0, init_state=None, duration=1000):
    if init_state == None:
        init_state = getQ(config, p, zid, jtm_idxs)
    
    if isinstance(jangs, torch.Tensor):  # Check if input is already a PyTorch tensor
        target_state = jangs
    else:
        target_state = torch.tensor(jangs).double()

    jt_idxs = jtm_idxs + [grip_idx]
    for i in range(duration):
        q = init_state*(1-i/duration) + target_state*(i/duration)# set position
        gripperQ = torch.tensor([gripper_target*(i/duration)]).double()
        q = torch.cat((q.reshape((6,)), gripperQ))

        qd = (target_state-init_state)/(duration*config.dt) # set velocity
        gripperQd = torch.tensor([0.1]).double() #-1*(i/duration)
        qd = torch.cat((qd.reshape((6,)), gripperQd))

        p.setJointMotorControlArray(zid, jt_idxs, p.POSITION_CONTROL, targetPositions=q,
                                    targetVelocities=qd)
        p.stepSimulation()
        time.sleep(config.dt)

def pause_and_calm(config, p, zid, jtm_idxs, grip_idx, arm_model, duration=200, init_pos=None):
    jt_idxs = jtm_idxs + [grip_idx]
    if init_pos == None:
        last_pos = getQ(config, p, zid, jt_idxs)
    else:
        last_pos = init_pos
    zero_vel = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    for i in range(0, duration):
        p.setJointMotorControlArray(zid, jt_idxs, p.POSITION_CONTROL, targetPositions=last_pos,
                                    targetVelocities=zero_vel)
        p.stepSimulation()
        time.sleep(config.dt)

def back_to_start(config, p, zid, jtm_idxs, grip_idx, arm_model, 
                  duration=10, to_start_duration=100):
    initq = [config.initq[i] for i in jtm_idxs]
    to_jangs_grip(torch.tensor(initq), config, p, zid, jtm_idxs, grip_idx, arm_model, 
                  gripper_target=0.0,duration=to_start_duration)
    last_pos = getQ(config, p, zid, jtm_idxs)
    zero_vel = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    gripperQ = torch.tensor([0.0]).double()
    q = torch.cat((last_pos.reshape((6,)), gripperQ))
    gripperQd = torch.tensor([0.0]).double()
    qd = torch.cat((zero_vel.reshape((6,)), gripperQd))
    
    jt_idxs = jtm_idxs + [grip_idx]
    for i in range(duration):
        p.setJointMotorControlArray(zid, jt_idxs, p.POSITION_CONTROL, targetPositions=q,
                                    targetVelocities=qd)
        p.stepSimulation()
        time.sleep(config.dt)

"""
The following code is for testing this single module, 
and can serve as a template for function usage.  
"""
if __name__ ==  "__main__":
    sys.path.append("../../z1_sdk/lib")
    import unitree_arm_interface

    print("-"*10+"Here we go"+"-"*10)

    np.set_printoptions(precision=3, suppress=True)
    arm_z1 = unitree_arm_interface.ArmInterface(hasGripper=True)
    arm_model = arm_z1._ctrlComp.armModel
    

    target_pos = [0.0, 0.20, -0.20, 0.48, 0.3, 0.40]
    target_pos = [3.00804, -0.66806,  0.44886, -0.16574, -0.26793,  0.50651]
    move_2p(target_pos, p, arm_model, const_vel=0.20)

    target_pos = [0.0, 0.20, 0.0, 0.48, 0.0, 0.20]
    move_2p(target_pos, p, arm_model, const_vel=0.20)