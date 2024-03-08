import os
from utils.motion_cam_images import generate_images
from observation_poses_config import ExperimentConfigs


exp_config = ExperimentConfigs()

"""
list of experiment tags and their saved image folder names (same means same name as tag):
2back_2side_leftback3linearx : same, 
2back_2sidesym_leftback3linearx : 2b2sslb3_block, 
2backsym_2sidesym_leftback3linearx : 2bs2sslb3_block,
"4back3sym_2sidesym_leftback3linearx" : 4bs2sslb3_block

"""
experiment_tag = "4back3sym_2sidesym_leftback3linearx"
pose_data = exp_config.get_config(experiment_tag)


# Running the arm to get pictures in assigned poses
images_saving_name = "4bs2sslb3_block"
saving_dir = f"arm_captured_images/{images_saving_name}"
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)
generate_images(end_effector_angles=pose_data.poses, tg_gripper_angs=pose_data.grippper_angs, 
                experiment_name=images_saving_name, conti_move_idxs=pose_data.conti_move_idxs,
                save_format=pose_data.image_format, saving_dir=saving_dir)