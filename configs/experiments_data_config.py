import os, sys
print("configuration file located at: ", os.path.dirname(os.path.realpath(__file__)))
project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_folder)
import re
from configs.observation_poses_config import ExperimentConfigs

class ArmDustrExpData():
    def __init__(self):
        self.obs_angles_config = ExperimentConfigs()
        self.dust3r_cam_poses_file_suffix = '_cam_poses.pt'
        self.ptc_file_suffix = '_ptc.pt'
        self.img_size = 512
        self.dust3r_local_dir = os.path.join(project_folder, "dust3r")
        self.standard_model_pth = (self.dust3r_local_dir + 
                                   "/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
        
        # Key: Name of each experiment -> Val: the corresponding experiment observation angles 
        # config when adding experiment, first add configuration in obs
        self.exp_name2configname = {
            '4bs2sslb3_sa': '4back3sym_2sidesym_leftback3linearx_sa', # (12+2 images)
            '4bs2sslb3_sa_apriltag': '4back3sym_2sidesym_leftback3linearx_sa', # same angle with apriltag
            '2bs2sslb3_sa': '2backsym_2sidesym_leftback3linearx_sa', #(8+2 images for < 10GB GPU)
            '2bs2sslb3_sa_apriltag': '2backsym_2sidesym_leftback3linearx_sa',

            # For comparison with colmap (14 + 2 images)
            '2obj_divangs' : 'diverse_ori_sa',
            '2obj_4cluster': 'fourcluster_ori_sa', 
            '2obj_backonly': 'backonly_ori_sa',

            '8obj_divangs' : 'diverse_ori_sa',
            '8obj_4cluster': 'fourcluster_ori_sa', 
            '8obj_backonly': 'backonly_ori_sa', 

            '7obj_divangs' : 'diverse_ori_sa',
            '7obj_4cluster': 'fourcluster_ori_sa', 
            '7obj_backonly': 'backonly_ori_sa', 
            
            # center of april tag about 0.15 m from base center
            'apriltag_divangs' : 'diverse_ori_sa',
            'apriltag_4cluster': 'fourcluster_ori_sa',
            'apriltag_backonly': 'backonly_ori_sa',
            
            'shelf_divangs' : 'shelf_div_sa',
            'shelf_4cluster': 'shelf_4cl_sa',
            'shelf_backonly': 'backonly_ori_sa',


            'apriltag_divangsa' : 'shelf_div_sa',
            'apriltag_4clustera': 'shelf_4cl_sa',
            'apriltag_backonlya': 'backonly_ori_sa',
        }

        self.expnames = self.exp_name2configname.keys()
        self.imgs_folder_pth = os.path.join(project_folder, "arm_captured_images")
        self.init_out_dir()

    def init_out_dir(self):
        self.dustr_out_pth = os.path.join(project_folder, "output/dust3r_saved_output")
        self.depth_out_pth = os.path.join(project_folder, "output/dc_saved_output")
        self.colmap_out_pth = os.path.join(project_folder, "output/colmap_saved_output")

    # Return the experiment angle configuration object, see configs/observation_poses_config.py
    def get_obs_config(self, experiment_name):
        if experiment_name not in self.expnames:
            raise Exception('Experiment configuration not found in configs/experiments_data_config.py')
        return self.obs_angles_config.get_config(self.exp_name2configname[experiment_name])
    
    # Return image dir path (image_type = 0 -> train_image, 1 -> test_image)
    def get_images_dir(self, exp_name):
        return os.path.join(self.imgs_folder_pth, exp_name)
    
    # Return list of image paths
    # If image_type=0 then they are sorted so their order is corresponded to the angles 
    # configuration order in configs/observation_poses_config.py.
    def get_images_paths(self, exp_name, for_colmap=False, num_imgs=None):
        file_paths = get_file_paths(os.path.join(self.imgs_folder_pth, exp_name))
        pose_data = self.get_obs_config(exp_name)
        file_paths_train = [file_paths[i] for i in range(len(file_paths)) if i not in pose_data.test_pt]
        print("total_traning_imgs:", len(file_paths_train))
        if num_imgs != None:
            file_paths_train = file_paths_train[:num_imgs]
            print(f"Training on {len(file_paths_train)} images")
        else:
            if not for_colmap:
                file_paths_train = [f for f in file_paths_train if 'cm' not in f] 
        return file_paths_train
    
    # Return the standard saved output paths (image_type = 0 -> arm_captured, 1 -> depth, 2 -> colemap)
    def get_ptc_output_path(self, exp_name, exp_type=0):
        file_name = exp_name+self.ptc_file_suffix
        if exp_type == 0:
            return os.path.join(self.dustr_out_pth, file_name)
        else: # dir
            return os.path.join(self.colmap_out_pth, exp_name)
        
    def get_cam_pose_path(self, exp_name):
        return os.path.join(self.dustr_out_pth, exp_name+self.dust3r_cam_poses_file_suffix)

def get_file_paths(folder_path):
    file_paths = []  # List to store file paths
    # Walk through all files and directories in the specified folder
    all_files_and_dirs = os.listdir(folder_path)
    
    # Filter out directories, keeping only files
    for f in all_files_and_dirs:
        path = os.path.join(folder_path, f)
        if os.path.isfile(path):
            file_paths.append(path)
    # Sort the list based on filenames
    file_paths.sort()
    return file_paths