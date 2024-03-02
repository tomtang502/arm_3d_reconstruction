import pybullet as p
import time
import pybullet_data
from robot_descriptions import z1_description
from robot_descriptions.loaders.pybullet import load_robot_description
from joint_info import *

def info_prt(s, name = ''):
    print("_"*60)
    if len(name) == 0:
        print(s)
    else:
        print(f"{name} : {s}")
    print("-"*60)


physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
print(p.getConnectionInfo())
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-9.8)
planeId = p.loadURDF("plane.urdf")
#planeId = load_robot_description("z1_description")
startPos = [0,0,0.5]
startOrientation = p.getQuaternionFromEuler([0,0,0])




"""
Begin Simulation
"""
print("-" * 20 + "Begin" + "-" * 20)
# set the center of mass frame (loadURDF sets base link frame) 
# startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
zid = p.loadURDF(z1_description.URDF_PATH, startPos, startOrientation)
num_joints = p.getNumJoints(zid)
jt_idxs = [i for i in range(num_joints)]
print(f"z1 has {num_joints} joints")

for i in range(num_joints):
    joint_inf_prt(zid, i)
print(p.POSITION_CONTROL)

for i in range (100000):
    p.setJointMotorControlArray(zid, jt_idxs, p.VELOCITY_CONTROL, targetVelocities = [1.0, 1.0, 0.5, 0.5, 0.3, 0.2, 5])

    p.stepSimulation()
    time.sleep(1./240.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(zid)



"""
End Simulation
"""
print("-" * 20 + "Simulation Ended" + "-" * 20)
print(cubePos,cubeOrn)
p.disconnect()




