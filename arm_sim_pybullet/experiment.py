import pybullet as p
import pybullet_data
import time, sys
from sim_utils.print_format import *

project_relative_path = "../"
sys.path.append(project_relative_path)
from utils.arm_motion_planning import *
from teaching_config import *


config = TeachingConfig(project_relative_path)

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
print(p.getConnectionInfo())
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-9.8)
#p.setTimeStep(0.002)
planeId = p.loadURDF("plane.urdf")
startPos = [0,0,0]
startOrientation = p.getQuaternionFromEuler([0,0,0])
# set the center of mass frame (loadURDF sets base link frame) 
# startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
zid = p.loadURDF(config.urdf_loc, startPos, startOrientation)
num_joints = p.getNumJoints(zid)
jt_idxs = [i for i in range(num_joints)]
print(f"z1 has {num_joints} joints")

"""
Begin Simulation
"""
print("-" * 20 + "Begin" + "-" * 20)

# for i in range(num_joints):
#     joint_inf_prt(zid, i)

# Sample joint angles (ensure this tensor is on the same device and dtype as your limits)
init_state = torch.tensor([-0.001, 0.006, -0.031, -0.079, -0.002, 0.001])
# from unitree as unitree format
init_pos = torch.tensor([0.0, -0.0755, 0.0, 0.0864, 0.0, 0.1778])
target_pos = torch.tensor([0.0, 0.20, 0.0, 0.48, 0.0, 0.20])
# convert format
init_pos = torch.cat((init_pos[-3:], init_pos[:3]))
target_pos = torch.cat((target_pos[-3:], target_pos[:3]))

link_name = config.end_effector_name
chain = load_chain_from_urdf(config.urdf_loc, link_name)

jangs_vel_list, jangs_pos_list = plan_motion(chain, init_state, init_pos, target_pos, dt = 1./240., 
                                             pos_diff_epsilon=0.0018, const_vel=0.10)
base_vel = torch.tensor([0.0])
for i in range (len(jangs_pos_list)):
    cur_jts_vel = torch.cat((base_vel, jangs_vel_list[i].reshape((6,)), base_vel, base_vel))
    p.setJointMotorControlArray(zid, jt_idxs, p.VELOCITY_CONTROL, 
                                targetVelocities = cur_jts_vel)

    p.stepSimulation()
    time.sleep(1./240.)

print(p.getLinkState(zid, 6))
for i in range(10000):
    #p.stepSimulation()
    time.sleep(1./240.)



"""
End Simulation
"""
print("-" * 20 + "Simulation Ended" + "-" * 20)
p.disconnect()




