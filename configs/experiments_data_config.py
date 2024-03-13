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
            '4bs3sslb3_sa': '4back3sym_3sidesym_leftback3linearx_sa', # (14+2 images)
            '4bs3sslb3_sa_apriltag': '4back3sym_3sidesym_leftback3linearx_sa',
            '2bs2sslb3_sa': '2backsym_2sidesym_leftback3linearx_sa', #(8+2 images for < 10GB GPU)
            '2bs2sslb3_sa_apriltag': '2backsym_2sidesym_leftback3linearx_sa'
        }

        self.expnames = self.exp_name2configname.keys()
        self.init_image_dir()
        self.init_out_dir()

    def init_image_dir(self):
        self.arm_imgs_folder_pth = os.path.join(project_folder, "captured_images/arm_captured_images")
        self.depth_imgs_folder_pth = os.path.join(project_folder, "captured_images/depth_images")
        self.colmap_imgs_folder_pth = os.path.join(project_folder, "captured_images/colmap_images")

    def init_out_dir(self):
        self.dustr_out_pth = os.path.join(project_folder, "output/dust3r_saved_output")
        self.depth_out_pth = os.path.join(project_folder, "output/dc_saved_output")
        self.colmap_out_pth = os.path.join(project_folder, "output/colmap_saved_output")

    # Return the experiment angle configuration object, see configs/observation_poses_config.py
    def get_obs_config(self, experiment_name):
        if experiment_name not in self.expnames:
            raise Exception('Experiment configuration not found in configs/experiments_data_config.py')
        return self.obs_angles_config.get_config(self.exp_name2configname[experiment_name])
    
    # Return list of image paths (image_type = 0 -> arm_captured, 1 -> depth, 2 -> colemap)
    # If image_type=0 then they are sorted so their order is corresponded to the angles 
    # configuration order in configs/observation_poses_config.py.
    def get_images_paths(self, exp_name, image_type=0):
        if image_type == 0:
            return get_file_paths(os.path.join(self.arm_imgs_folder_pth, exp_name), 
                                  key_function=extract_number_with_extension)
        elif image_type == 1:
            return get_file_paths(os.path.join(self.depth_imgs_folder_pth, exp_name))
        else:
            return get_file_paths(os.path.join(self.colmap_imgs_folder_pth, exp_name))
    
    # Return the standard saved output paths (image_type = 0 -> arm_captured, 1 -> depth, 2 -> colemap)
    def get_ptc_output_path(self, exp_name, exp_type=0):
        file_name = exp_name+self.ptc_file_suffix
        if exp_type == 0:
            return os.path.join(self.dustr_out_pth, file_name)
        elif exp_type == 1:
            return os.path.join(self.depth_out_pth, file_name)
        else:
            return os.path.join(self.colmap_out_pth, file_name)
        
    def get_cam_pose_path(self, exp_name):
        return os.path.join(self.dustr_out_pth, exp_name+self.dust3r_cam_poses_file_suffix)
    
def extract_number_with_extension(file_path):
    """Extracts the numerical suffix from a file path, ignoring the file extension."""
    match = re.search(r'_([0-9]+)\.[a-zA-Z]+$', file_path)
    if match:
        return int(match.group(1))
    return -1  # In case there's no numerical suffix

def get_file_paths(folder_path, key_function=lambda x: x):
    file_paths = []  # List to store file paths
    # Walk through all files and directories in the specified folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)  # Create full file path
            file_paths.append(file_path)  # Add file path to list
    # Sort the list based on filenames
    file_paths.sort(key=key_function)
    return file_paths[:12]