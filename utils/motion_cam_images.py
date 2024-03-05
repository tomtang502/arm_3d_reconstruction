import sys, time, cv2
from arm_motion import to_jangs, pause_and_calm, back_to_start
sys.path.append("../z1_sdk/lib")
import unitree_arm_interface
import numpy as np

experiment_name = "sixangs_1feature"
save_format = 'jpg'
print("Press ctrl+\ to quit process.")

"""
top hand cam
right 3.008 -0.668 0.449 -0.16574 -0.268 0.5065 (from start)
left -3.008 -0.5914 -0.633 -0.111  0.2634  0.485 (from start)
x+5cm -3.008 -0.5914 -0.633 -0.061  0.2634  0.485
x+10cm -3.008 -0.5914 -0.633 -0.011  0.2634  0.485

r_side -2.3205  0.1745 -0.2795  0.291 -0.5694  0.357 (from start)
l_side 2.341 0.252 0.145 0.466 0.4645 0.3435 (from start)
x-5cm: 2.341 0.252 0.145 0.416 0.4645 0.3435
"""
# roll pitch yaw x y z in meter
top_cam_cposes = [
    [3.008, -0.668, 0.549, -0.16574, -0.268, 0.5065], # right back
    [-3.008, -0.5914, -0.633, -0.111, 0.2634, 0.485], # left back
    [-3.008, -0.5914, -0.633, -0.061, 0.2634, 0.485], # left back x + 5cm
    [-3.008, -0.5914, -0.633, -0.011, 0.2634, 0.485], # left back x + 10cm
    [-1.8205, 0.179, -0.2795, 0.256, -0.5694, 0.157],
    [1.721, 0.179, 0.2195, 0.326, 0.5645, 0.1535]
]
# Gripper angle in radians
tg_gripper_angs = [
    -1.5,
    -1.5,
    -1.5,
    -1.5,
    -np.pi/2 + 0.001,
    -np.pi/2 + 0.001
]

start_loc = [-0.001, 0.006, -0.031, -0.079, -0.002, 0.001]
top_pos = []
conti_move_idxs = [1, 2]

# Open handles to the webcams


np.set_printoptions(precision=3, suppress=True)
arm =  unitree_arm_interface.ArmInterface(hasGripper=True)
# arm_model = arm._ctrlComp.armModel
armState = unitree_arm_interface.ArmFSMState
arm.loopOn()
arm.backToStart()
total_num_images = len(top_cam_cposes)
for i in range(total_num_images):
    print(f"Start Taking {i + 1} out of {total_num_images}")
    jnt_speed = 1.0
    arm.MoveJ(np.array(top_cam_cposes[i]), tg_gripper_angs[i], jnt_speed)
    cam = cv2.VideoCapture(2)
    cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, -4.25)  # Example value
    success_captured, img = cam.read()
    
    if success_captured:
        saved_name = f'{experiment_name}_{i}.{save_format}'
        cv2.imwrite(f'../arm_captured_images/{saved_name}', img)
        print(saved_name, "saved!")   
    else:
        print(f"unable to capture frame {i}")
    cam.release()
    if i not in conti_move_idxs:
        arm.backToStart()

arm.loopOff()

cv2.destroyAllWindows()