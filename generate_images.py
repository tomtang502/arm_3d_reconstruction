import os
from utils.motion_cam_images import generate_images
from configs.observation_poses_config import ExperimentConfigs


exp_config = ExperimentConfigs()

experiment_tag = "xyz3linear_5back_2sidesym_sa"
pose_data = exp_config.get_config(experiment_tag)


# Running the arm to get pictures in assigned poses
images_saving_name = "xyz3each_16imgs_sa"
saving_dir = f"arm_captured_images/{images_saving_name}"
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)
generate_images(end_effector_angles=pose_data.poses, tg_gripper_angs=pose_data.grippper_angs, 
                experiment_name=images_saving_name, conti_move_idxs=pose_data.conti_move_idxs,
                save_format=pose_data.image_format, saving_dir=saving_dir, cam_idx=0)