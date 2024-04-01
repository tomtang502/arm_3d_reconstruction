import torch
import numpy as np
from dt_apriltags import Detector
import cv2
import os
import copy
from utils.geometric_util import *
from utils.graph_util import *
from utils.fix_scale_calib import *
from configs.experiments_data_config import ArmDustrExpData
from scipy.spatial.transform import Rotation as R
from utils.scale_calib import *

exp_config = ArmDustrExpData()
                       
def get_file_paths(folder_path):
    file_paths = []  # List to store file paths
    # Walk through all files and directories in the specified folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)  # Create full file path
            file_paths.append(file_path)  # Add file path to list
    # Sort the list based on filenames
    block = ['arm_captured_images/2bs2sslb3_sa/2bs2sslb3_sa_0.jpg'] 
    # Used to block out noisy apriltag
    file_paths.sort()
    # 10, 15 for apriltag_4clustera
    # 8 for apriltag_backonly with tag_id == 1
    # 2, 11, 12 for apriltag_backonlya tag_id == 2
    # file_paths = (file_paths[0:2] + 
    #               block +
    #               file_paths[3:11] + 
    #               block * 2 +
    #               file_paths[13:24]
    #               )
    file_paths = file_paths[:24]
    #file_paths = file_paths[:4] +['arm_captured_images/2bs2sslb3_sa/2bs2sslb3_sa_0.jpg']+ file_paths[5:]
    print(len(file_paths))
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
    
    rotation = R.from_matrix(rotation_difference)

    # Convert to Euler angles, specifying the axes sequence
    euler_angles = rotation.as_euler('XYZ')
    angle_difference = torch.norm(torch.tensor(euler_angles))

    # Combine translation and rotation differences into a single scalar loss
    #loss = translation_difference + angle_difference
    return translation_difference, angle_difference

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

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    at_detector = Detector(families='tag36h11',
                        nthreads=1,
                        quad_decimate=1.0,
                        quad_sigma=0.0,
                        refine_edges=1,
                        decode_sharpening=0.25,
                        debug=0)


    
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

    print("tag showed frqeuency:", frequency)
    selected_tag_id = max(frequency.keys(), key=lambda key:frequency[key])
    #selected_tag_id = 2
    print("tag_id selected:", selected_tag_id)   

    
    ground_truth_poses = [] # size 24 (20 + 4, since cuda out of memory)
    gt_poses_idx = []
    for i in range(len(file_paths)):
        gt_single_pose = tags_cent_list[i].get(selected_tag_id, None)
        if isinstance(gt_single_pose, np.ndarray):
            gt_poses_idx.append(i)
            gt_single_pose = (torch.tensor(gt_single_pose))
            ground_truth_poses.append(gt_single_pose) 

    print(f"Ground truth has size {len(ground_truth_poses)}")
    assert len(ground_truth_poses) == frequency[selected_tag_id], "Number of tags used != Number of cam poses!"
    print(f"Apriltag detected idx {gt_poses_idx}")
    ground_truth_poses = np.array(ground_truth_poses)
    print(ground_truth_poses.shape)
    
    ground_truth_poses = np.linalg.pinv(ground_truth_poses)

    # xyz_gt = tmatw2c_to_xyz([-torch.linalg.pinv(tmat) for tmat in ground_truth_poses])
    # xyz_gt = torch.stack(xyz_gt)
    # xyz_R = ground_truth_poses[:, :3, :3]
    # xyz_RT = np.transpose(xyz_R, [0,2,1])
    # xyz = ground_truth_poses[:, :3, -1]
    # print(xyz_RT.shape)
    # print(xyz.shape)
    # rot_y = np.array([[-1,  0,  0],
    #                   [ 0,  1,  0],
    #                   [ 0,  0, -1]])
    # for i in range(xyz.shape[0]):
        # xyz[i] = xyz_RT[i]@xyz[i]
    
    #graph_double_struct(ground_truth_poses[:, :3, -1], eef_nontest)

    exp_name = "8obj_divangs"
    pose_data = exp_config.get_obs_config(exp_name)
    eef_poses_all = pose_data.poses + pose_data.additional_colmap_pose
    eef_poses_tor=pose_to_transform(torch.tensor(eef_poses_all))

    eef_sc_used, apriltag_sc_used, eef_nontest = scale_calib_pose_process_col(eef_poses_tor, 
                                                                            ground_truth_poses, 
                                                                            gt_poses_idx, 
                                                                            [], 
                                                                            pose_data.linearidx)
    graph_double_struct(ground_truth_poses[:, :3, -1], eef_nontest[:,:3,3])
    print(eef_nontest.shape, "should be same as", ground_truth_poses.shape)
    assert eef_nontest.shape == ground_truth_poses.shape, "Number of eef != Number of cam poses!"

    print(eef_sc_used.shape, "should be same (sc) as", apriltag_sc_used.shape)

    T = caculate_calib_trans_mat(eef_sc_used, apriltag_sc_used)
    ground_truth_poses_sc = T.float()@ground_truth_poses


    
    # Visualize constructed ptc
    cam_pos_n=ground_truth_poses_sc[:,:3,3]
    eff_poses_n=eef_nontest[:,:3,3]
    plotty_graph_multistruct([eff_poses_n, cam_pos_n], 
                             ["arm end-effector", "cam_pose"],
                             [2, 2])



    tensors_to_save = {
        'poses': ground_truth_poses_sc,
        'dense_pt': None,
        'eef_poses': eef_nontest,
        'eef_idx': gt_poses_idx,
        'T' : T
    }

    # Saving the dictionary of tensors to a file
    saving_loc = os.path.join("output/apriltag_saved_output", f'{expap_name}.pth')
    torch.save(tensors_to_save, saving_loc)
    print("="*10)
    print(f"april_tag out saved at {saving_loc}")
    print("="*10)



# # graph_double_struct(xyz_gt, xyz_eef)
