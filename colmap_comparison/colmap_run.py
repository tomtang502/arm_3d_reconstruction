import pycolmap, torch, pathlib, os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import numpy as np
import open3d as o3d
import shutil
import utils.geometric_util as geomu
from utils.graph_util import graph_single_struct, graph_double_struct, plotty_graph_multistruct
from utils.scale_calib import *
from utils.fix_scale_calib import transpose_poses_ptc
from configs.experiments_data_config import ArmDustrExpData
exp_config = ArmDustrExpData()


exp_name = '8obj_divangs'
num_imgs = 10
#writing_file = 'output/colmap_calib_loss.txt'

def copy_images_to_tmp(original_folder, idxs, parent_folder, n_imgs):
    """
    Copy specified images from the original folder to a temporary folder under the specified parent folder.

    Args:
    original_folder: Path to the original folder containing the images.
    parent_folder: Path to the parent folder where the temporary folder should be created.

    Returns:
    Path to the temporary folder where the images are copied.
    """
    # Create a temporary directory under the parent folder
    tmp_folder = os.path.join(parent_folder, "tmp")
    os.makedirs(tmp_folder, exist_ok=True)

    i, cnt = 0, 0 
    filenames = os.listdir(original_folder)
    filenames.sort()
    # Copy images to the temporary folder
    for filename in filenames:
        if (filename.endswith(('.jpg', '.jpeg', '.png', '.gif')) and
            i not in idxs):
            original_path = os.path.join(original_folder, filename)
            if os.path.isfile(original_path):
                shutil.copy(original_path, tmp_folder)
                cnt += 1
        i += 1
        if cnt >= n_imgs:
            break

    return tmp_folder

def delete_tmp_folder(tmp_folder):
    """
    Delete the temporary folder.

    Args:
    tmp_folder: Path to the temporary folder to delete.
    """
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)

# Example usage:

output_path = pathlib.Path(exp_config.get_ptc_output_path(exp_name, exp_type=1))
print("Colmap saving folder at", output_path)
if not os.path.exists(output_path):
    original_folder = exp_config.get_images_dir(exp_name)
    pose_data = exp_config.get_obs_config(exp_name)
    tmp_folder = copy_images_to_tmp(original_folder, pose_data.test_pt, "output", num_imgs)
    # Copy images to the temporary folder under the parent folder
    print("Images copied to temporary folder:", tmp_folder)
    print(output_path)
    image_dir = pathlib.Path(tmp_folder)

    output_path.mkdir()
    mvs_path = output_path / "mvs"
    database_path = output_path / "database.db"

    pycolmap.extract_features(database_path, image_dir)#, sift_options={"max_num_features": 512})
    #pycolmap.extract_features(database_path, image_dir)
    pycolmap.match_exhaustive(database_path)
    maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
    maps[0].write(output_path)

    pycolmap.undistort_images(mvs_path, output_path, image_dir)
    pycolmap.patch_match_stereo(mvs_path)  # requires compilation with CUDA
    pycolmap.stereo_fusion(output_path / "dense.ply", mvs_path)


    # Delete the temporary folder
    delete_tmp_folder(tmp_folder)
    print("Temporary folder deleted.")


"""
Running Caiberation
"""
reconstruction = pycolmap.Reconstruction(output_path)
pose_data = exp_config.get_obs_config(exp_name)
num_additional = num_imgs - len(pose_data.poses) + len(pose_data.test_pt)
eef_poses = pose_data.poses + pose_data.additional_colmap_pose
eef_poses_tor=geomu.pose_to_transform(torch.tensor(eef_poses))

col_cam_poses = []
selected_idx = []
idx_map = dict()
col_cam_poses_map = dict()
i = 0
for image_id, image in reconstruction.images.items():
    name = image.name[:-len('.jpg')].split('_')[2]
    idx = 0
    if len(name) == 1:
        idx = ord(name) - ord('a')
    else:
        idx = 26 + ord(name[1]) - ord('a')
    selected_idx.append(idx)
    idx_map[idx] = i
    i += 1
    img_pose = np.array(image.cam_from_world.matrix())
    pose_tmat = torch.tensor(geomu.colmap_pose2transmat(img_pose))
    col_cam_poses.append(pose_tmat)
    col_cam_poses_map[idx] = pose_tmat.clone()

combined = list(zip(selected_idx, col_cam_poses))
combined.sort(key=lambda x:x[0])
selected_idx = [x[0] for x in combined]
col_cam_poses = [x[1] for x in combined]

print("colmap selected:", selected_idx)

ply_path = os.path.join(exp_config.get_ptc_output_path(exp_name, exp_type=1), "dense.ply")
point_cloud = o3d.io.read_point_cloud(ply_path)
ptc_xyz = np.asarray(point_cloud.points)
# o3d.visualization.draw_geometries([point_cloud])
if point_cloud.colors:
    # Extract color information
    ptc_colors = np.asarray(point_cloud.colors)
else:
    ptc_colors = None
    print("This point cloud has no color information.")


ptc_tor = torch.tensor(ptc_xyz)
poses_tor = torch.tensor(np.stack(col_cam_poses))
im_poses_tor_o = torch.linalg.pinv(poses_tor)

eef_sc_used, colmap_sc_used, eef_nontest = scale_calib_pose_process_col(eef_poses_tor, 
                                                                            im_poses_tor_o, 
                                                                            selected_idx, 
                                                                            pose_data.test_pt, 
                                                                            pose_data.linearidx)
xyz = np.stack(geomu.tmatw2c_to_xyz(im_poses_tor_o))
xyz_eef = np.stack(geomu.tmatw2c_to_xyz(eef_nontest))
#graph_double_struct(xyz_eef, xyz)
assert eef_nontest.shape == im_poses_tor_o.shape, "Number of eef != Number of cam poses!"


### Solving for scale and then do caliberation
T, scale, J, R_L, t_L  = compute_arm(eef_sc_used, colmap_sc_used, colmap=True)
im_poses_tor_o[:,:3,3]=im_poses_tor_o[:,:3,3]*scale
ptc_tor_o = ptc_tor*scale


loss_info = f'{exp_name}_{num_imgs} trans loss: {t_L.mean()}, rot loss: {R_L.mean()}\n'

print(loss_info)
# with open(writing_file, 'a') as file:
#     file.write(loss_info) 

colmap_pose, colmap_ptc = transpose_poses_ptc(im_poses_tor_o.float(), ptc_tor_o.float(), T)


#Visualize constructed ptc
pts_tor_n = colmap_ptc[::10]
cam_pos_n=colmap_pose[:,:3,3]
eff_poses_n=eef_nontest[:,:3,3]
plotty_graph_multistruct([eff_poses_n, cam_pos_n, pts_tor_n], 
                         ["arm end-effector", "camera pose", "point cloud"],
                         [2, 2, 0.3])


tensors_to_save = {
    'poses': colmap_pose,
    'dense_pt': colmap_ptc,
    'eef_poses': eef_nontest,
    'T' : T,
    'J' : torch.tensor(J),
    'trans_L' : torch.tensor(t_L),
    'rot_L' : torch.tensor(R_L)
}


# Saving the dictionary of tensors to a file
saving_loc = os.path.join("output/colmap_saved_output", f'{exp_name}_{num_imgs}.pth')
torch.save(tensors_to_save, saving_loc)
print("="*10)
print(f"colmap out saved at {saving_loc}")
print("="*10)