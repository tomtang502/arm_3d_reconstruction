import os
from utils.motion_cam_images import generate_images
from configs.experiments_data_config import ArmDustrExpData

exp_config = ArmDustrExpData()
exp_name = "4obj_measure"

pose_data = exp_config.get_obs_config(exp_name)
web_cam_idx = 2

# Running the arm to get pictures in assigned poses
saving_dir = exp_config.get_images_dir(exp_name)
print(f"Images files saved at {saving_dir}")
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)


generate_images(end_effector_angles=pose_data.poses, 
                colmapimg_angs=[],
                tg_gripper_angs=pose_data.grippper_angs,
                comap_griang=pose_data.colmap_gripper_ang,
                experiment_name=exp_name, conti_move_idxs=pose_data.conti_move_idxs,
                save_format=pose_data.image_format, saving_dir=saving_dir, cam_idx=web_cam_idx)
