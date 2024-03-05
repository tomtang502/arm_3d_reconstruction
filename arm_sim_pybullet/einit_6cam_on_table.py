import pybullet as p
import pybullet_data as pdata
import time, sys, os, torch
from sim_utils.print_format import *
from sim_utils.sim_arm_motion import move_2p_grip, to_jangs_grip, back_to_start, pause_and_calm

project_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."
sys.path.append(project_dir)
sys.path.append(project_dir + "/z1_sdk/lib")
import unitree_arm_interface
from teaching_config import *
config = TeachingConfig(project_dir)

config.dt = 1./240.
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
# print(p.getConnectionInfo())
p.setGravity(0,0,-9.8)
p.setTimeStep(config.dt)
p.setAdditionalSearchPath(pdata.getDataPath())

"""
Setting up onjects
"""
print("-" * 20 + "Loading Components" + "-" * 20)
# -- ground --
planeId = p.loadURDF("plane.urdf")
# -- arm --
startPos = [-0.5,0,0.6] # in center of mass frame
startOrientation = p.getQuaternionFromEuler([0,0,0])
zid = p.loadURDF(config.urdf_loc, startPos, startOrientation)
num_joints = p.getNumJoints(zid)
jt_idxs = [i for i in range(num_joints)]
jt6_idxs = [1,2,3,4,5,6]
grip_idx = 8
# -- table --
startPos = [0,0,0] # in center of mass frame
startOrientation = p.getQuaternionFromEuler([0,0,0])
tabid = p.loadURDF("table/table.urdf", startPos, startOrientation, useFixedBase=True)
print(p.getLinkState(tabid, 3))

"""
Begin Simulation
"""
print("-" * 20 + "Begin" + "-" * 20)

init_state = torch.tensor([-0.001, 0.006, -0.031, -0.079, -0.002, 0.001]).double()
# from unitree as unitree format
target_pos = [0.0, 0.20, 0., 0.48, 0.0, 0.20]

top_gripper_clocs = [
    [0.866679, 0.226716, -2.174512, 1.220503, -0.319004, 2.738409],
    [-1.011894, 0.164913, -1.998049, 1.191000, 0.312839, -2.792527],
    [-1.296526, 2.477420, -2.595039, 0.443646, 0.995403, -2.595000],
    [0.940528, 2.477420, -2.595039, 0.474628, -0.768597, 2.592034]
]
gripper_ang = [-1, -1, -3, -3]
arm = unitree_arm_interface.ArmInterface(hasGripper=True)
arm_model = arm._ctrlComp.armModel
#move_2p_grip(target_pos, config, p, zid, jt6_idxs, grip_idx, arm_model, gripper_target=-1.0)


start_loc = [-0.001, 0.006, -0.031, -0.079, -0.002, 0.001]
"""
-0.06296  0.28114 -0.10987  0.03558  0.02220  0.19199
-0.03597  0.53740 -0.22307  0.01029  0.08084  0.33757

top hand cam
right [3.00804, -0.66806,  0.44886, -0.16574, -0.26793,  0.50651]
left -3.00884 -0.59141 -0.63270 -0.11136  0.26341  0.48496
r_side -2.32052  0.17450 -0.27954  0.29085 -0.56943  0.35672
l_side 2.34120 0.25178 0.14510 0.46627 0.46455 0.34350
"""


total_num_images = len(top_gripper_clocs)
for i in range(total_num_images):
    print(f"Start Taking {i + 1} out of {total_num_images}")
    to_jangs_grip(top_gripper_clocs[i], config, p, zid, jt6_idxs, grip_idx, arm_model, 
                  gripper_target=gripper_ang[i], duration=500)
    pose_maintained = torch.cat((torch.tensor(top_gripper_clocs[i]).reshape((6,)).double(),
                                 torch.tensor([gripper_ang[i]]).double()))
    pause_and_calm(config, p, zid, jt6_idxs, grip_idx, arm_model, duration=200, 
                   init_pos=pose_maintained)
    back_to_start(config, p, zid, jt6_idxs, grip_idx, arm_model, 
                  duration=10, to_start_duration=500)


#print(p.getLinkState(zid, 7))
for i in range(10000):
    p.stepSimulation()
    time.sleep(config.dt)



"""
End Simulation
"""
print("-" * 20 + "Simulation Ended" + "-" * 20)
p.disconnect()



