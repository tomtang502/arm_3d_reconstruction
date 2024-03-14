import sys, os, cv2
project_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."
sys.path.append(project_dir)
from z1_sdk.lib import unitree_arm_interface
import numpy as np

# Open handles to the webcams

def generate_images(end_effector_angles, colmapimg_angs, tg_gripper_angs, comap_griang, 
                    experiment_name, conti_move_idxs, save_format, saving_dir, cam_idx=2):
    assert len(end_effector_angles) == len(tg_gripper_angs)
    print("Press ctrl+\ to quit process.")
    np.set_printoptions(precision=3, suppress=True)
    arm =  unitree_arm_interface.ArmInterface(hasGripper=True)
    # arm_model = arm._ctrlComp.armModel
    armState = unitree_arm_interface.ArmFSMState
    arm.loopOn()
    arm.backToStart()
    num_ori = len(end_effector_angles)
    total_num_images = num_ori + len(colmapimg_angs)
    for i in range(total_num_images):
        print(f"Start Taking {i + 1} out of {total_num_images}")
        jnt_speed = 1.0
        if i >= num_ori:
            tg_pose, tg_griang = np.array(colmapimg_angs[i-num_ori]), comap_griang
        else:
            tg_pose, tg_griang = np.array(end_effector_angles[i]), tg_gripper_angs[i]
        arm.MoveJ(tg_pose, tg_griang, jnt_speed)
        cam = cv2.VideoCapture(cam_idx)
        cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, -4.25)  # Example value
        success_captured, img = cam.read()
        
        if success_captured:
            letter_idx = chr(ord('a') + i)
            if i >= num_ori:
                saved_name = f'{experiment_name}_{letter_idx}_cm.{save_format}'
            else:
                saved_name = f'{experiment_name}_{letter_idx}.{save_format}'
            cv2.imwrite(f'{saving_dir}/{saved_name}', img)
            print(saved_name, "saved!")   
        else:
            print(f"unable to capture frame {i}")
        cam.release()
        if i not in conti_move_idxs:
            arm.backToStart()

    arm.loopOff()

    cv2.destroyAllWindows()