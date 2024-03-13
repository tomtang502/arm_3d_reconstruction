import numpy as np
class ObsAngsConfig():
    def __init__(self, poses, grippper_angs, experiment_name, 
                 linear_idx=dict(), conti_move_idxs=None, image_format='jpg'):
        self.poses = poses
        self.grippper_angs = grippper_angs
        self.name = experiment_name
        if conti_move_idxs == None:
            self.conti_move_idxs = []
        else:
            self.conti_move_idxs = conti_move_idxs
        self.image_format = image_format
        self.linearidx = linear_idx# dict of x, y, z linear motion
        

class ExperimentConfigs():
    def __init__(self, rel_path=""):
        """
        Absolute initial position of the gripper loading plane (the J6 end plane): 
            [0,049, 0, 0.1605]
        The position of the center of the Unitree_gripper relative to the loading plane: 
            [0.0382, 0, 0].
        End-effector pose (positiona + orientation) should be recorded as 
            [roll, pitch, yaw, x, y, z] in meter
        Gripper angle should be a single float value
        """
        self.exps = dict()
        self.start_pose = [-0.001, 0.006, -0.031, -0.079, -0.002, 0.001]
        # Adding different experimental angles
        self.fourback3sym_twosidesym_leftback3linearx_sa()
        self.fourback3sym_threesidesym_leftback3linearx_sa()
        self.twobacksym_twosidesym_leftback3linearx_sa()
        self.xyz2linear_twobacksym_threeside_sa()
        self.xyz3linear_fiveback_twosidesym_sa()
        self.threebacksym_twosidesym_lb3lx_sa()
    
    def get_config(self, name):
        if name in self.exps:
            return self.exps[name]
        else:
            raise Exception(f"The experiment with name {name} is not defined in observation_poses_config.py yet!")

    def add_experiment(self, experiment_name, poses, gripper_angs, linear_idx=dict(),
                       conti_move_idxs=None, image_format='jpg'):
        self.exps[experiment_name] = ObsAngsConfig(poses, gripper_angs, experiment_name, 
                                                   linear_idx, conti_move_idxs, image_format)

    # ----------------- example of adding an experiment -----------------
    # Don't fogot to call this in your python file or in __init__!
    def fourback3sym_twosidesym_leftback3linearx_sa(self):
        experiment_name = "4back3sym_2sidesym_leftback3linearx_sa"
        # roll pitch yaw x y z in meter
        top_cam_cposes = [
            [3.008, -0.668, 0.549, -0.16574, -0.268, 0.5065], # right back 0
            [3.008, -0.5914, 0.633, -0.111, -0.2634, 0.485], # right back 1
            [-2.31092, -0.24056, -0.99504, -0.21117, -0.12188, 0.41125], #right back 2
            [-2.51890, -0.54126, -0.08807, -0.00074, -0.49342, 0.42817], # right back corner
            [-3.008, -0.668, -0.549, -0.16574, 0.268, 0.5065], # left back 0
            [-3.008, -0.5914, -0.633, -0.111, 0.2634, 0.485], # left back 1
            [-3.008, -0.5914, -0.633, -0.061, 0.2634, 0.485], # left back 1 x + 5cm
            [-3.008, -0.5914, -0.633, -0.011, 0.2634, 0.485], # left back 1 x + 10cm
            [2.31092, -0.24056, 0.99504, -0.21117, 0.12188, 0.41125], # left back 2
            [2.39645, -0.54843, 0.08445, -0.00101, 0.49278, 0.42958], # left back corner

            [-1.8205, 0.179, -0.2795, 0.256, -0.5694, 0.157], # left side 0
            [-1.721, 0.179, -0.2195, 0.326, -0.5645, 0.1535], # left side 1
            [1.721, 0.179, 0.2195, 0.326, 0.5645, 0.1535], # right side 0
            [1.8205, 0.179, 0.2795, 0.256, 0.5694, 0.157] # right side 1
        ]
        # Gripper angle in radians
        tg_gripper_angs = [-np.pi/2 + 0.001] * len(top_cam_cposes)
        conti_move_idxs = [0, 1, 2, 4, 5, 6, 7, 8, 10, 12]
        additional_linear_pts = [5, 6]
        linear_idx = dict()
        linear_idx['x'] = (additional_linear_pts+[7], additional_linear_pts,  0.05)

        self.add_experiment(experiment_name=experiment_name, poses=top_cam_cposes, 
                            gripper_angs=tg_gripper_angs, linear_idx=linear_idx,
                            conti_move_idxs=conti_move_idxs)
      
    def fourback3sym_threesidesym_leftback3linearx_sa(self):
        experiment_name = "4back3sym_3sidesym_leftback3linearx_sa"
        # roll pitch yaw x y z in meter
        top_cam_cposes = [
            [3.008, -0.668, 0.549, -0.16574, -0.268, 0.5065], # right back 0
            [3.008, -0.5914, 0.633, -0.111, -0.2634, 0.485], # right back 1
            [-2.31092, -0.24056, -0.99504, -0.21117, -0.12188, 0.41125], #right back 2
            [-2.51890, -0.54126, -0.08807, -0.00074, -0.49342, 0.42817], # right back corner
            [-3.008, -0.668, -0.549, -0.16574, 0.268, 0.5065], # left back 0
            [-3.008, -0.5914, -0.633, -0.111, 0.2634, 0.485], # left back 1
            [-3.008, -0.5914, -0.633, -0.061, 0.2634, 0.485], # left back 1 x + 5cm
            [-3.008, -0.5914, -0.633, -0.011, 0.2634, 0.485], # left back 1 x + 10cm
            [2.31092, -0.24056, 0.99504, -0.21117, 0.12188, 0.41125], # left back 2
            [2.39645, -0.54843, 0.08445, -0.00101, 0.49278, 0.42958], # left back corner

            [1.721, 0.179, 0.2195, 0.326, 0.5645, 0.1535], # left side 0
            [1.8205, 0.179, 0.2795, 0.256, 0.5694, 0.157], # left side 1
            [2.28403, 0.05687, 0.21994, 0.40305, 0.46293, 0.37193], # left side 2
            [-1.8205, 0.179, -0.2795, 0.256, -0.5694, 0.157], # right side 0
            [-1.721, 0.179, -0.2195, 0.326, -0.5645, 0.1535], # right side 1
            [-2.28403, 0.05687, -0.21994, 0.40305, -0.46293, 0.37193] # right side 2
        ]
        # Gripper angle in radians
        tg_gripper_angs = [-np.pi/2 + 0.001] * len(top_cam_cposes)
        conti_move_idxs = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 13, 14]
        additional_linear_pts = [5, 6]
        linear_idx = dict()
        # For one axis, element at idx 7 (e7) is the original point, and the additional_linear_pts 
        # are points that is obtained from moving parallel on an axis from e7
        linear_idx['x'] = (additional_linear_pts+[7], additional_linear_pts,  0.05)
        

        self.add_experiment(experiment_name=experiment_name, poses=top_cam_cposes, 
                            gripper_angs=tg_gripper_angs, linear_idx=linear_idx, 
                            conti_move_idxs=conti_move_idxs)
        
    def twobacksym_twosidesym_leftback3linearx_sa(self):
        experiment_name = "2backsym_2sidesym_leftback3linearx_sa"
        # roll pitch yaw x y z in meter
        top_cam_cposes = [
            [3.008, -0.668, 0.549, -0.16574, -0.268, 0.5065], # right back 0
            [-2.51890, -0.54126, -0.08807, -0.00074, -0.49342, 0.42817], # right back corner
            [-3.008, -0.5914, -0.633, -0.111, 0.2634, 0.485], # left back 1
            [-3.008, -0.5914, -0.633, -0.061, 0.2634, 0.485], # left back 1 x + 5cm
            [-3.008, -0.5914, -0.633, -0.011, 0.2634, 0.485], # left back 1 x + 10cm
            [2.31092, -0.24056, 0.99504, -0.21117, 0.12188, 0.41125], # left back 2
          
            [1.8205, 0.179, 0.2795, 0.256, 0.5694, 0.157], # left side 1
            [2.28403, 0.05687, 0.21994, 0.40305, 0.46293, 0.37193], # left side 2
            [-1.8205, 0.179, -0.2795, 0.256, -0.5694, 0.157], # right side 0
            [-2.28403, 0.05687, -0.21994, 0.40305, -0.46293, 0.37193] # right side 2   
        ]
        # Gripper angle in radians
        tg_gripper_angs = [-np.pi/2 + 0.001] * len(top_cam_cposes)
        conti_move_idxs = [0, 2, 3, 4, 6, 8]
        additional_linear_pts = [2, 3]
        linear_idx = dict()
        linear_idx['x'] = (additional_linear_pts+[4], additional_linear_pts, 0.05)


        self.add_experiment(experiment_name=experiment_name, poses=top_cam_cposes, 
                            gripper_angs=tg_gripper_angs, linear_idx=linear_idx,
                            conti_move_idxs=conti_move_idxs)
        
    def xyz3linear_fiveback_twosidesym_sa(self):
        experiment_name = "xyz3linear_5back_2sidesym_sa"
        # roll pitch yaw x y z in meter
        top_cam_cposes = [
            [3.008, -0.668, 0.549, -0.16574, -0.268, 0.5065], # right back 0
            [-2.51890, -0.54126, -0.08807, -0.00074, -0.49342, 0.42817], # right back corner
            [-3.008, -0.5914, -0.633, -0.111, 0.2634, 0.485], # left back 1
            [2.31092, -0.24056, 0.99504, -0.21117, 0.12188, 0.41125], # left back 2

            [-2.98462, -0.59539, -0.63417, -0.02,  0.15,  0.55564], # new y-10cm
            [-2.98462, -0.59539, -0.63417, -0.02,  0.20,  0.55564], # new y-5cm 
            [-2.98462, -0.59539, -0.63417, -0.02,  0.25,  0.55564], # new
            [-2.98462, -0.59539, -0.63417, 0.03,  0.25,  0.55564], # new x+5cm
            [-2.98462, -0.59539, -0.63417, 0.08,  0.25, 0.55564], # new x+10cm
            [-2.98462, -0.59539, -0.63417, -0.02,  0.25,  0.55564], # new
            [-2.98462, -0.59539, -0.63417, -0.02,  0.25,  0.60564], # new z+5cm
            [-2.98462, -0.59539, -0.63417, -0.02,  0.25,  0.50564], # new z-5cm
            
        
            [1.8205, 0.179, 0.2795, 0.256, 0.5694, 0.157], # left side 1
            [2.28403, 0.05687, 0.21994, 0.40305, 0.46293, 0.37193], # left side 2
            [-1.8205, 0.179, -0.2795, 0.256, -0.5694, 0.157], # right side 0
            [-2.28403, 0.05687, -0.21994, 0.40305, -0.46293, 0.37193] # right side 2   
        ]
        # Gripper angle in radians
        tg_gripper_angs = [-np.pi/2 + 0.0001] * len(top_cam_cposes)
        conti_move_idxs = [0] + [2, 3, 4, 5, 6, 7, 8, 9, 10] + [12, 14]
        self.add_experiment(experiment_name=experiment_name, poses=top_cam_cposes, 
                            gripper_angs=tg_gripper_angs, conti_move_idxs=conti_move_idxs)

    def xyz2linear_twobacksym_threeside_sa(self):
        experiment_name = "xyz2linear_2backsym_3side_sa"
        # roll pitch yaw x y z in meter
        top_cam_cposes = [
            [3.008, -0.668, 0.549, -0.16574, -0.268, 0.5065], # right back 0
            [-2.51890, -0.54126, -0.08807, -0.00074, -0.49342, 0.42817], # right back corner
            [2.31092, -0.24056, 0.99504, -0.21117, 0.12188, 0.41125], # left back 2

            [-2.98462, -0.59539, -0.63417, -0.02,  0.15,  0.55564], # new y-10cm
            [-2.98462, -0.59539, -0.63417, -0.02,  0.25,  0.55564], # new
            [-2.98462, -0.59539, -0.63417, 0.08,  0.25, 0.55564], # new x+10cm
            [-2.98462, -0.59539, -0.63417, -0.02,  0.25,  0.45564], # new z-10cm
            
        
            [1.8205, 0.179, 0.2795, 0.256, 0.5694, 0.157], # left side 0
            [-1.8205, 0.179, -0.2795, 0.256, -0.5694, 0.157], # right side 0
            [-2.28403, 0.05687, -0.21994, 0.40305, -0.46293, 0.37193] # right side 1  
        ]
        # Gripper angle in radians
        tg_gripper_angs = [-np.pi/2 + 0.0001] * len(top_cam_cposes)
        conti_move_idxs = [0] + [2, 3, 4, 5] + [8]
        self.add_experiment(experiment_name=experiment_name, poses=top_cam_cposes, 
                            gripper_angs=tg_gripper_angs, conti_move_idxs=conti_move_idxs)
        
    def threebacksym_twosidesym_lb3lx_sa(self):
        experiment_name = "3backsym_2sidesym_lb3lx_sa"
        # roll pitch yaw x y z in meter
        top_cam_cposes = [
            [3.008, -0.668, 0.549, -0.16574, -0.268, 0.5065], # right back 0
            [3.008, -0.5914, 0.633, -0.111, -0.2634, 0.485], # right back 1
            [-2.51890, -0.54126, -0.08807, -0.00074, -0.49342, 0.42817], # right back corner
            [-3.008, -0.5914, -0.633, -0.111, 0.2634, 0.485], # left back 1
            [-3.008, -0.5914, -0.633, -0.061, 0.2634, 0.485], # left back 1 x + 5cm
            [-3.008, -0.5914, -0.633, -0.011, 0.2634, 0.485], # left back 1 x + 10cm
            [2.31092, -0.24056, 0.99504, -0.21117, 0.12188, 0.41125], # left back 2
            [2.39645, -0.54843, 0.08445, -0.00101, 0.49278, 0.42958], # left back corner

            [1.721, 0.179, 0.2195, 0.326, 0.5645, 0.1535], # left side 0
            [2.28403, 0.05687, 0.21994, 0.40305, 0.46293, 0.37193], # left side 2
            [-1.8205, 0.179, -0.2795, 0.256, -0.5694, 0.157], # right side 0
            [-2.28403, 0.05687, -0.21994, 0.40305, -0.46293, 0.37193] # right side 2   
        ]
        # Gripper angle in radians
        tg_gripper_angs = [-np.pi/2 + 0.001] * len(top_cam_cposes)
        conti_move_idxs = [0, 1, 3, 4, 5, 6, 8, 10]

        self.add_experiment(experiment_name=experiment_name, poses=top_cam_cposes, 
                            gripper_angs=tg_gripper_angs, conti_move_idxs=conti_move_idxs)