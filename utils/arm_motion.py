import sys, time
import torch
sys.path.append("../z1_sdk/lib")
import unitree_arm_interface
import numpy as np
from arm_motion_planning import plan_motion, joint_to_pose, pose_shape

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

# move from one point to another point
def move_2p(p_f, arm, arm_model, const_vel=0.10):

    init_state = torch.tensor(arm.lowstate.getQ()).double()
    
    #from unitree as unitree format (row, pitch, yaw, x, y, z)
    p_i = joint_to_pose(init_state, arm_model.forwardKinematics)
    init_pos = torch.tensor(p_i.reshape((pose_shape[0],))).double()
    target_pos = torch.tensor(p_f).double()

    _, jangs_pos_list, _ = plan_motion(None, init_state, init_pos, target_pos, const_vel=const_vel, 
                                       arm_Jacobian=arm_model.CalcJacobian, 
                                       forward_kin=arm_model.forwardKinematics)
    
    init_state = init_state.numpy()
    arm.setFsmLowcmd()
    target_state = jangs_pos_list[-1]
    
    duration = len(jangs_pos_list)
    for i in range(duration):
        arm.q = init_state*(1-i/duration) + target_state*(i/duration)# set position
        arm.qd = (target_state-init_state)/(duration*0.002) # set velocity
        # arm.q = jangs_pos_list[i] # set position
        # arm.qd = jangs_vel_list[i] # set velocity
        arm.tau = arm_model.inverseDynamics(arm.q, arm.qd, np.zeros(6), np.zeros(6)) # set torque
        arm.gripperQ = 0 #-1*(i/duration)
        arm.setArmCmd(arm.q, arm.qd, arm.tau)
        arm.setGripperCmd(arm.gripperQ, arm.gripperQd, arm.gripperTau)
        arm.sendRecv()# udp connection
        # print(arm.lowstate.getQ())
        time.sleep(arm._ctrlComp.dt)


"""
The following code is for testing this single module, 
and can serve as a template for function usage.  
"""
if __name__ ==  "__main__":
    

    print("-"*10+"Here we go"+"-"*10)

    np.set_printoptions(precision=3, suppress=True)
    arm = unitree_arm_interface.ArmInterface(hasGripper=True)
    arm_model = arm._ctrlComp.armModel
    arm.setFsmLowcmd()

    target_pos = [0.0, 0.20, -0.20, 0.48, 0.3, 0.40]
    move_2p(target_pos, arm, arm_model, const_vel=0.10)

    target_pos = [0.0, 0.20, 0.0, 0.48, 0.0, 0.20]
    move_2p(target_pos, arm, arm_model, const_vel=0.10)

    #input("Press the Enter key to continue: ") 
    arm.loopOn()
    arm.backToStart()
    arm.loopOff()
    