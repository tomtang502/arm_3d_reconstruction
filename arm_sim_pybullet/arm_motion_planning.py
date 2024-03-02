import torch
import numpy as np
import pytorch_kinematics as pk
import math, sys, os
from robot_descriptions import z1_description


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
"""
joint_angle_shape = (6, 1)
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

# init_pos, target_pos are row vector tensor of (x, y, z)
def ee_velocity(init_pos, target_pos, t):
    return (target_pos - init_pos) / t

"""
init_pos, tar_pos: column vectors in (x, y, z, roll, pitch, yaw)
init_state: a row vector from top to bottom in the order of from joint angle 
closest to the base (J1) to joint angle closest to the end-effector (J6).
# Assume input state is valid, which is within bound

"""
def plan_motion(chain, init_state, init_pos, target_pos, dt=0.002, const_vel = 0.10, 
                joint_angle_shape=joint_angle_shape):
    init_pos = convert_to_tensor(init_pos) 
    tar_pos = convert_to_tensor(target_pos).reshape(joint_angle_shape)
    # all following operations done in tensor
    cur_pos = init_pos.reshape(joint_angle_shape)
    cur_state = torch.clamp(init_state, joint_min, joint_max).reshape(joint_angle_shape)
    keep_move = True
    jangs_vel_list = list()
    jangs_pos_list = list()

    # start planning
    
    while (keep_move):
        #
        #Add some check here to terminate

        dist = torch.norm(tar_pos - cur_pos)
        t = dist/const_vel
        ee_vel = ee_velocity(cur_pos, tar_pos, t)
        
        J = chain.jacobian(cur_state.reshape(joint_angle_shape[0]))
        J = J.squeeze(0)
        J_inv = torch.pinverse(J)

        jangs_vel = torch.matmul(J_inv, ee_vel.reshape((J_inv.shape[0], 1)))
        jangs_vel_list.append(jangs_vel)
        
        cur_state = jangs_vel.reshape(joint_angle_shape) * dt + cur_state
        cur_state = torch.clamp(cur_state, joint_min, joint_max)
        jangs_pos_list.append(cur_state)
        cur_pos = torch.matmul(J, jangs_vel) + cur_pos
        
    return jangs_vel_list, jangs_pos_list

if __name__ ==  "__main__":
    # Sample joint angles (ensure this tensor is on the same device and dtype as your limits)
    const_vel = 0.10
    cur_angles = torch.tensor([-math.pi / 2.0, 100, math.pi, 0.0, math.pi / 2.0, 0.0]).reshape(joint_angle_shape)

    # Clamp joint angles to respect joint limits
    
    cur_angles_c = torch.clamp(cur_angles, joint_min, joint_max)
    print(cur_angles_c)



    link_name = "link06"
    chain = load_chain_from_urdf(z1_description.URDF_PATH, link_name)
    #chain = pk.build_serial_chain_from_urdf(open(z1_description.URDF_PATH).read(), link_name)

    # Use the clamped joint angles for kinematics computations
    print("here")
    J = chain.jacobian(cur_angles_c.reshape(joint_angle_shape[0]))
    J = J.squeeze(0)

    print(J.shape)
    J_inv = torch.pinverse(J)
    print(J_inv.shape)
