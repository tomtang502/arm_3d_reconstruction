import sys, time
from arm_motion import to_jangs
sys.path.append("../z1_sdk/lib")
import unitree_arm_interface
import numpy as np

print("Press ctrl+\ to quit process.")

back_gripper_clocs = [
    [0.866679, 0.226716, -2.174512, 1.220503, -0.319004, 2.738409],
    [-3.00884, -0.59141, -0.63270, -0.11136,  0.26341,  0.48496],
    [0.13717, 0.43749, 0.32348, 0.05619, -0.14178, 0.22141],
    [0.07096, 0.52266, 0.05814, -0.04354, -0.06640, 0.33880]
]
"""
-0.06296  0.28114 -0.10987  0.03558  0.02220  0.19199
-0.03597  0.53740 -0.22307  0.01029  0.08084  0.33757

top hand cam
right [3.00804, -0.66806,  0.44886, -0.16574, -0.26793,  0.50651]
left -3.00884 -0.59141 -0.63270 -0.11136  0.26341  0.48496
r_side -2.32052  0.17450 -0.27954  0.29085 -0.56943  0.35672
l_side 2.34120 0.25178 0.14510 0.46627 0.46455 0.34350
"""

np.set_printoptions(precision=3, suppress=True)
arm =  unitree_arm_interface.ArmInterface(hasGripper=True)
arm_model = arm._ctrlComp.armModel
arm.setFsmLowcmd()


to_jangs(arm, arm_model, back_gripper_clocs[0])

# for pt in back_gripper_clocs:
#     gripper_pos = 0.0
#     print(pt)
#     arm.labelRun("top_cam_rb")
arm.loopOn()
arm.backToStart()
arm.loopOff()

