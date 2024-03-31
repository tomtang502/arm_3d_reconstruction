

"""
Pre-processing
"""
import torch
import numpy as np
from dt_apriltags import Detector
import cv2
import os
import copy
from utils.geometric_util import *
from utils.graph_util import *
from utils.scale_calib import *
from configs.experiments_data_config import ArmDustrExpData

exp_config = ArmDustrExpData()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)
                       
def get_file_paths(folder_path):
    file_paths = []  # List to store file paths
    # Walk through all files and directories in the specified folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)  # Create full file path
            file_paths.append(file_path)  # Add file path to list
    # Sort the list based on filenames
    file_paths.sort()
    return file_paths

def combine_matrices(rot_matrix, translation_matrix):
    """
    Combine a 3x3 transformation matrix and a 1x3 translation matrix into a 4x4 transformation matrix.

    Args:
    transformation_matrix: 3x3 numpy array representing the transformation matrix.
    translation_matrix: 1x3 numpy array representing the translation matrix.

    Returns:
    combined_matrix: 4x4 numpy array representing the combined transformation matrix.
    """
    # Create a 4x4 identity matrix
    combined_matrix = np.eye(4)
    
    # Copy the transformation matrix into the top-left 3x3 portion of the combined matrix
    combined_matrix[:3, :3] = rot_matrix
    
    # Copy the translation matrix into the last column of the combined matrix
    combined_matrix[:3, 3] = translation_matrix.squeeze()
    
    return combined_matrix

def matrix_loss(matrix1, matrix2):
    """
    Calculate the scalar loss between two transformation matrices.

    Args:
    matrix1, matrix2: Numpy arrays representing the 4x4 transformation matrices.

    Returns:
    Scalar loss/error between the matrices.
    """
    # Calculate translation difference
    translation1 = matrix1[:, 3]
    translation2 = matrix2[:, 3]
    translation_difference = np.linalg.norm(translation1 - translation2)

    # Calculate rotation difference
    rotation1 = matrix1[:3, :3]
    rotation2 = matrix2[:3, :3]
    rotation_difference = torch.linalg.pinv(rotation1) * rotation2
    angle_difference = torch.norm(euler_angles = torch.nn.functional.rotation_matrix_to_euler(rotation_difference, 'XYZ'))

    # Combine translation and rotation differences into a single scalar loss
    loss = translation_difference + angle_difference
    return loss




    
focal_length = 727
img_size =(640,480)
center = (img_size[0]/2, img_size[1]/2)
mtx=np.array([[focal_length, 0., center[0]],
                           [0., focal_length, center[1]],
                           [0., 0., 1.]])
camera_params_c=[focal_length,focal_length,center[0],center[1]]

dist_g= 0.065
width_s = 2.
height_s = 3.
width_np=np.arange(0,width_s)
height_np=np.arange(0,height_s)
w, h = np.meshgrid(width_np, height_np, indexing='ij')
grid=np.concatenate([w[:,:,None],h[:,:,None]],2).reshape((-1,2))
t_grid=copy.copy(grid)
t_grid[:,0]=((grid[:,0]+0.5)-0.5*width_s)*dist_g
t_grid[:,1]=((grid[:,1]+0.5)-0.5*height_s)*dist_g
t_grid_3d=np.concatenate([t_grid,np.zeros((len(t_grid),1))],axis=1)
distortion_coeffs = np.array([])





# ----------------------------------------------- #
###################### Main #######################
# ----------------------------------------------- #
expap_name = "apriltag_divangs"
linear_idxs = [5, 6, 7]
remove_idx = [25, 26, 27]
x_d = 0.05
folder_path = f'arm_captured_images/{expap_name}'
file_paths = get_file_paths(folder_path)

tags_cent_list= dict([(i, dict()) for i in range(len(file_paths))])
frequency = dict()
for idx in range(len(file_paths)):
    file_str=file_paths[idx]
    img = cv2.imread(file_str, cv2.IMREAD_GRAYSCALE)
    tags = at_detector.detect(img, estimate_tag_pose=True,
                            camera_params=camera_params_c,
                            tag_size=0.05)
    for i in range(len(tags)):
        id = tags[i].tag_id
        frequency[id] = 1 + frequency.get(id, 0)
        tags_cent_list[idx][id] = combine_matrices(tags[i].pose_R,tags[i].pose_t)

def find_common_element(dict1, dict2, dict3, priority):
    """
    Find one of the common integer elements in three dictionaries with the highest priority.

    Args:
    dict1, dict2, dict3: Dictionaries to search for common elements.
    priority: List of integers representing the priority of integers to search for.

    Returns:
    An integer common to all three dictionaries with the highest priority, or None if not found.
    """
    sorted_priority = sorted(priority.items(), key=lambda x: x[1], reverse=True)
    for num, p in sorted_priority:
        if num in dict1 and num in dict2 and num in dict3:
            return num
    return None
    
selected_pose = find_common_element(tags_cent_list[5], tags_cent_list[6], tags_cent_list[7], frequency)
selected_pose_alter = find_common_element(tags_cent_list[25], tags_cent_list[26], tags_cent_list[27], frequency)
if selected_pose == None:
    selected_pose = selected_pose_alter

ground_truth_poses = [] # size 28
for i in range(len(file_paths)):
    gt_sigle_pose = tags_cent_list[i].get(selected_pose, None)
    if isinstance(gt_sigle_pose, np.ndarray):
        gt_sigle_pose = torch.linalg.pinv(torch.tensor(gt_sigle_pose))
    ground_truth_poses.append(gt_sigle_pose) 

print(f"Ground truth always has size {len(ground_truth_poses)}")

gt_idxs = [i for i in range(len(ground_truth_poses)) if isinstance(ground_truth_poses[i], torch.Tensor)]
print(gt_idxs)
exp_name = "shelf_divangs"

xyz_gt = tmatw2c_to_xyz([torch.linalg.pinv(tmat) for tmat in ground_truth_poses if isinstance(tmat, torch.Tensor)])
xyz_gt = torch.stack(xyz_gt)

for i in linear_idxs:
    if not isinstance(ground_truth_poses[i], torch.Tensor):
        print("no! caliberation problem")
        exit(1)

pose_data = exp_config.get_obs_config(exp_name)
eef_poses_vec = pose_data.poses + pose_data.additional_colmap_pose
eef_poses_tor=pose_to_transform(torch.tensor(eef_poses_vec))

gt_poses_o = []
gt_poses_oa = []
eef_poses_o = []
eef_poses_oa = []
for i in range(len(ground_truth_poses)):
    gd_tmat = ground_truth_poses[i]
    if isinstance(gd_tmat, torch.Tensor):
        gt_poses_oa.append(gd_tmat)
        eef_poses_oa.append(eef_poses_tor[i])
        if (i not in linear_idxs) and (i not in remove_idx):
            gt_poses_o.append(gd_tmat)
            eef_poses_o.append(eef_poses_tor[i])
    
gt_poses_o = torch.stack(gt_poses_o)
eef_poses_o = torch.stack(eef_poses_o)
gt_poses_oa = torch.stack(gt_poses_oa)
eef_poses_oa = torch.stack(eef_poses_oa)
print(len(gt_poses_o), len(eef_poses_o))

scale_factor = get_scale_factor_col(ground_truth_poses, linear_idxs, x_d)
gt_poses_o[:,:3,3]=gt_poses_o[:,:3,3]*scale_factor
gt_poses_oa[:,:3,3]=gt_poses_oa[:,:3,3]*scale_factor
#gt_poses_oas = rescale_pose_tag(ground_truth_poses, gt_poses_oa, linear_idxs, x_d)

# R, t, _, _ = cv2.calibrateRobotWorldHandEye(gt_poses_os[:, :3, :3].numpy(), gt_poses_os[:, :3, 3].numpy(),
#                                eef_poses_o[:, :3, :3].numpy(), eef_poses_o[:, :3, 3].numpy(), 
#                                method = cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH )
# T = torch.zeros((4, 4)).float()
# T[3, 3] = 1.
# T[:3, :3] = torch.tensor(R)
# print(t)
# T[:3, 3] = torch.tensor(t.reshape((3,)))
# T = torch.linalg.pinv(T)
T = caculate_calib_trans_mat(eef_poses_o, gt_poses_o)
gt_poses = T.float()@gt_poses_oa.float()

xyz_gt = np.stack(tmatw2c_to_xyz(gt_poses))
xyz_eef = np.stack(tmatw2c_to_xyz(eef_poses_oa))
j = 0
for i in range(len(ground_truth_poses)):
    if isinstance(ground_truth_poses[i], torch.Tensor):
        ground_truth_poses[i] = gt_poses[j]
        j += 1
xyz_gt2 = np.stack(tmatw2c_to_xyz(torch.stack([t for t in ground_truth_poses if isinstance(t, torch.Tensor)]).numpy()))
graph_double_struct(xyz_eef, xyz_gt2, ['xyz_eef', 'xyz_gt2'])