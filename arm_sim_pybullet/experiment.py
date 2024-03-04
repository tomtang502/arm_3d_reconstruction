import pybullet as p
import pybullet_data
import time, sys
from sim_utils.print_format import *

project_relative_path = "../"
sys.path.append(project_relative_path)
from utils.arm_motion_planning import *
from teaching_config import *
sys.path.append("../z1_sdk/lib")
import unitree_arm_interface


config = TeachingConfig(project_relative_path)

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
print(p.getConnectionInfo())
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-9.8)
dt = 1./240.
p.setTimeStep(dt)
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
init_state = torch.tensor([-0.001, 0.006, -0.031, -0.079, -0.002, 0.001]).double()
# from unitree as unitree format
init_pos = torch.tensor([0.0, -0.0755, 0.0, 0.0864, 0.0, 0.1778]).double()
target_pos = torch.tensor([0.0, 0.20, -0.20, 0.48, 0.3, 0.40]).double()

link_name = config.end_effector_name
chain = load_chain_from_urdf(config.urdf_loc, link_name)

arm = unitree_arm_interface.ArmInterface(hasGripper=True)
armModel = arm._ctrlComp.armModel
jangs_vel_list, jangs_pos_list, _ = plan_motion(chain, init_state, init_pos, target_pos, dt = 1./240., 
                                             pos_diff_epsilon=0.0018, const_vel=0.10, 
                                             to_numpy=False, arm_Jacobian=armModel.CalcJacobian,
                                             forward_kin=armModel.forwardKinematics)
base_vel = torch.tensor([0.0]).double()

init_state = init_state
target_state = jangs_pos_list[-1]
duration = len(jangs_pos_list)
for i in range(duration):
    q = init_state*(1-i/duration) + target_state*(i/duration)# set position
    gripperQ = torch.tensor([0.0]).double() #-1*(i/duration)
    q = torch.cat((base_vel, q.reshape((6,)), gripperQ, base_vel))

    qd = (target_state-init_state)/(duration*dt) # set velocity
    gripperQd = torch.tensor([0.1]).double() #-1*(i/duration)
    qd = torch.cat((base_vel, qd.reshape((6,)), gripperQd, base_vel))

    p.setJointMotorControlArray(zid, jt_idxs, p.POSITION_CONTROL, targetPositions=q,
                                targetVelocities=qd)
    p.stepSimulation()
    time.sleep(dt)

print(p.getLinkState(zid, 7))
for i in range(10000):
    p.stepSimulation()
    time.sleep(dt)



"""
End Simulation
"""
print("-" * 20 + "Simulation Ended" + "-" * 20)
p.disconnect()




