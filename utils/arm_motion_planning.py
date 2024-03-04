import torch
import numpy as np
import pytorch_kinematics as pk
from scipy.spatial.transform import Rotation as R
import math, sys, os
sys.path.append("../z1_sdk/lib")
import unitree_arm_interface



"""
Constraint on arm joints, starting from joint angle closest to the base (J1) to joint angle closest 
to the end-effector (J6). This is for unitree_z1
Lower:
[-math.pi*5/6, 
 0., 
 -math.pi*11/12, 
 -math.pi*4/9, 
 -math.pi*17/36, 
 -math.pi*8/9]

Upper:
[math.pi*5/6, 
 math.pi, 
 0., 
 math.pi*4/9, 
 math.pi*17/36, 
 math.pi*8/9]

GRIPPER IS NOT INCLUDED HERE
"""
joint_angle_shape = (6, 1)
pose_shape = (6, 1)
joint_min = torch.tensor([-2.6179938779914944, 
                          0., 
                          -2.8797932657906435, 
                          -1.5184364492350666, 
                          -1.3439035240356338, 
                          -2.792526803190927]).reshape(joint_angle_shape)
joint_max = torch.tensor([2.6179938779914944, 
                          2.9670597283903604, 
                          0., 
                          1.5184364492350666, 
                          1.3439035240356338, 
                          2.792526803190927]).reshape(joint_angle_shape)

"""
Functions to plan the joints angular velocities of the arm
"""
def convert_to_tensor(input_data):
    if isinstance(input_data, np.ndarray):  # Check if input is a NumPy array
        return torch.from_numpy(input_data)  # Convert NumPy array to PyTorch tensor
    elif isinstance(input_data, torch.Tensor):  # Check if input is already a PyTorch tensor
        return input_data
    else:
        raise ValueError("Input must be a NumPy array or a PyTorch tensor")

def load_chain_from_urdf(file_name, link_name, device = torch.device("cpu")):
    # Suppress stderr, the is for urdf that includes hardware description like z1 arm
    sys.stderr = open(os.devnull, 'w')
    chain = pk.build_serial_chain_from_urdf(open(file_name).read(), link_name)
    # Restore stderr
    sys.stderr = sys.__stderr__
    return chain

#forward_kin is a function that take in joint angles and output transfomation matrix
def joint_to_pose(jangs, forward_kin, pose_shape=pose_shape):
    T = forward_kin(jangs, 6); # this should come out as 4x4 matrix
    loc = T[:3, -1:]

    # Convert rotation matrix to Euler angles
    rot_mat = T[:3, :3]
    r = R.from_matrix(rot_mat)
    rot = r.as_euler('xyz', degrees=False)
    pose = np.concatenate((rot, loc.reshape(pose_shape[0]//2,))).reshape(pose_shape)
    return pose

"""
init_pos, tar_pos: column vectors in (roll, pitch, yaw, x, y, z)
init_state: a row vector from top to bottom in the order of from joint angle 
closest to the base (J1) to joint angle closest to the end-effector (J6).
# Assume input state is valid, which is within bound
# tmax should be in second
# pos_diff_epsilon is measured in m, and the motion terminates when the position within this range
  of the target position
# if arm_jacobian is None, then this function will use jacobian calculation in pytorch_kinematic,
  which will require float datatype
"""
def plan_motion(chain, init_state, init_pos, target_pos, dt=0.002, const_vel=0.10, 
                tmax=20, pos_diff_epsilon=0.0001, pose_shape=pose_shape, 
                joint_angle_shape=joint_angle_shape, 
                to_numpy=True, arm_Jacobian=None, forward_kin = None):
    if not ((chain != None) or ((arm_Jacobian != None) and (forward_kin != None))):
        raise Exception('You need to either provide pytorch_kinematic chain from urdf',
                         '(you can use load_chain_from_urdf to do this)',
                         'or you have to provide both arm_jacobian and pose_shape functions!')
    init_pos = convert_to_tensor(init_pos) 
    tar_pos = convert_to_tensor(target_pos).reshape(pose_shape)
    # all following operations done in tensor
    cur_pos = init_pos.reshape(pose_shape)
    cur_state = torch.clamp(init_state.reshape(joint_angle_shape), joint_min, joint_max).reshape(joint_angle_shape)
    jangs_vel_list = list()
    jangs_pos_list = list()

    # start planning
    iterations = round(tmax/dt)
    for i in range(iterations):
        dist = torch.norm(tar_pos - cur_pos)
        if dist < pos_diff_epsilon:
            print(f"term iters = {i}/{iterations}")
            break
        ee_vel = (tar_pos - cur_pos) / dist * const_vel
        
        if arm_Jacobian == None:
            J = chain.jacobian(cur_state)
            J = J.squeeze(0)
        else:
            J = torch.tensor(arm_Jacobian(cur_state))
        J_inv = torch.pinverse(J)

        jangs_vel = torch.matmul(J_inv, ee_vel.reshape((J_inv.shape[0], 1)))
        jangs_vel_list.append(jangs_vel.reshape((pose_shape[0],)))

        cur_state = jangs_vel.reshape(joint_angle_shape) * dt + cur_state
        cur_state = torch.clamp(cur_state, joint_min, joint_max) 
        jangs_pos_list.append(cur_state.reshape((joint_angle_shape[0],)))
        if forward_kin != None:
            cur_pos = joint_to_pose(cur_state, forward_kin)
        else:
            cur_pos = torch.matmul(J, jangs_vel) * dt + cur_pos
        
    if to_numpy:
        jangs_vel_list = [x.numpy() for x in jangs_vel_list]
        jangs_pos_list = [x.numpy() for x in jangs_pos_list]
    return jangs_vel_list, jangs_pos_list, cur_pos

"""
The following code is for testing this single module, 
and can serve as a template for function usage.  
"""
if __name__ ==  "__main__":
    
    # Sample joint angles (ensure this tensor is on the same device and dtype as your limits)
    init_state = torch.tensor([-0.001, 0.006, -0.031, -0.079, -0.002, 0.001]).double()
    # from unitree as unitree format
    init_pos = torch.tensor([0.0, -0.0755, 0.0, 0.0864, 0.0, 0.1778]).double()
    target_pos = torch.tensor([0.0, 0.20, 0.0, 0.48, 0.0, 0.20]).double()
    # convert format
    # init_pos = torch.cat((init_pos[-3:], init_pos[:3]))
    # target_pos = torch.cat((target_pos[-3:], target_pos[:3]))
    link_name = "gripperStator"
    chain = load_chain_from_urdf("z1_description/z1_gripper.urdf", link_name)
    
    arm = unitree_arm_interface.ArmInterface(hasGripper=True)
    armModel = arm._ctrlComp.armModel
    
    jangs_vel_list, jangs_pos_list, final_pose = plan_motion(chain, init_state, init_pos, 
                                                             target_pos, const_vel=0.10, 
                                                             arm_Jacobian=armModel.CalcJacobian,
                                                             forward_kin=armModel.forwardKinematics)
    print(jangs_pos_list[-1])
    print(final_pose)