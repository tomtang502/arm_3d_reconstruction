import numpy as np
class ObsAngsConfig():
    def __init__(self, poses, grippper_angs, experiment_name, linear_idx=dict(), 
                 conti_move_idxs=None, image_format='jpg', test_pt=[]):
        self.poses = poses
        self.grippper_angs = grippper_angs
        self.name = experiment_name
        if conti_move_idxs == None:
            self.conti_move_idxs = []
        else:
            self.conti_move_idxs = conti_move_idxs
        self.image_format = image_format
        self.linearidx = linear_idx # dict of x, y, z linear motion
        self.additional_colmap_pose = [
            [2.64908, -0.53835, 0.72063, -0.03105, 0.02311, 0.43604], # 16
            [-3.01590, -0.58084, -0.34939, 0.00479, 0.08198, 0.55425],
            [-3.07847, -0.69409, -0.62859, -0.06854,  0.26391, 0.48303], # 18
            [-3.02032, -0.75549, -0.90440, 0.00144, 0.20123, 0.58582],
            [-2.94671, -0.62757, -0.43736, -0.03666, 0.10900, 0.64760], # 20
            [-2.59892, -0.43580, -0.69952, 0.00325, -0.05990, 0.52683], 
            [2.59892, -0.43580, 0.69952, 0.00325, 0.05990, 0.52683], #22
            [-2.74252, -0.49285, -0.83710, -0.01153, 0.08183, 0.61475],
            [-1.721, 0.179, -0.2195, 0.326, -0.5645, 0.1535], #24

            [-2.98462, -0.59539, -0.63417, -0.02,  0.15,  0.55564], # new y
            [-2.98462, -0.59539, -0.63417, -0.02,  0.20,  0.55564], # new y + 5cm
            [-2.98462, -0.59539, -0.63417, -0.02,  0.25,  0.55564] # new y + 10cm
        ]
        self.colmap_gripper_ang = grippper_angs[0]
        self.test_pt = test_pt        

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
        self.twobacksym_twosidesym_leftback3linearx_sa()
        self.threebacksym_twosidesym_lb3lx_sa()
        # The orientations used for comparison with colmap
        self.diverse_ori_sa()
        self.fourcluster_ori_sa()
        self.backonly_ori_sa()
        self.shelf_4cl_sa()
        self.shelf_div_sa()
        self.scale_sa()

    
    def get_config(self, name):
        if name in self.exps:
            return self.exps[name]
        else:
            raise Exception(f"The experiment with name {name} is not defined in observation_poses_config.py yet!")

    def add_experiment(self, experiment_name, poses, gripper_angs, linear_idx=dict(),
                       conti_move_idxs=None, image_format='jpg', test_pt=[]):
        self.exps[experiment_name] = ObsAngsConfig(poses, gripper_angs, experiment_name, 
                                                   linear_idx, conti_move_idxs, image_format,
                                                   test_pt)

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
            [1.8205, 0.179, 0.2795, 0.256, 0.5694, 0.157] # right side 1test_pt=test_pt
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

    def diverse_ori_sa(self):
        experiment_name = "diverse_ori_sa"
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
            
            [-2.87854, -0.62008, 0.39347, 0.15808, -0.43670, 0.48861], # right cside
            [2.90352, -0.58821, 0.90963, 0.18867, -0.25014, 0.53657], # right side 0
            [-1.8205, 0.179, -0.2795, 0.256, -0.5694, 0.157], # right side 1

            [2.41919, -0.40692, 0.26378, 0.13205, 0.42530, 0.46276], # left cside
            [1.721, 0.179, 0.2195, 0.326, 0.5645, 0.1535], # left side 0
            [1.8205, 0.179, 0.2795, 0.256, 0.5694, 0.157] # left side 1
        ]
        # Gripper angle in radians
        tg_gripper_angs = [-np.pi/2 + 0.001] * len(top_cam_cposes)
        conti_move_idxs = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 13, 14]
        test_pt = [1, 4, 13, 14]
        linear_idx = [5, 6]

        self.add_experiment(experiment_name=experiment_name, poses=top_cam_cposes, 
                            gripper_angs=tg_gripper_angs, linear_idx=linear_idx, 
                            conti_move_idxs=conti_move_idxs, test_pt=test_pt)
    
    def fourcluster_ori_sa(self):
        experiment_name = "fourcluster_ori_sa"
        # roll pitch yaw x y z in meter
        top_cam_cposes = [
            [3.008, -0.668, 0.549, -0.16574, -0.268, 0.5065], # right back 0
            [3.008, -0.5914, 0.633, -0.111, -0.2634, 0.485], # right back 1
            [-2.23329, -0.26615, -1.04705, -0.19794, -0.12522, 0.42122], #right back 2
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
        
        linear_idx = [5, 6]
        test_pt = [0, 4, 10, 15]
        self.add_experiment(experiment_name=experiment_name, poses=top_cam_cposes, 
                            gripper_angs=tg_gripper_angs, linear_idx=linear_idx, 
                            conti_move_idxs=conti_move_idxs, test_pt=test_pt)
        
    def backonly_ori_sa(self):
        experiment_name = "backonly_ori_sa"
        # roll pitch yaw x y z in meter
        top_cam_cposes = [
            [3.008, -0.668, 0.549, -0.16574, -0.268, 0.5065], # right back 0
            [3.008, -0.5914, 0.633, -0.111, -0.2634, 0.485], # right back 1
            [-2.23329, -0.26615, -1.04705, -0.19794, -0.12522, 0.42122], #right back 2
            [2.92374, -0.61803, 0.39798, -0.19502, -0.17614, 0.57953], # right back 3
            [-2.82464, -0.56304, -0.19736, -0.04839, -0.20244, 0.48914], # right back 4
            [2.85661, -0.49566, 0.43731, -0.06788, -0.08179, 0.65387], # right back 5
            [2.70836, -0.36149, 0.79123, -0.00340, 0.01288, 0.59120], # cright back

            [-3.008, -0.668, -0.549, -0.16574, 0.268, 0.5065], # left back 0
            [-3.008, -0.5914, -0.633, -0.111, 0.2634, 0.485], # left back 1
            [-3.008, -0.5914, -0.633, -0.061, 0.2634, 0.485], # left back 1 x + 5cm
            [-3.008, -0.5914, -0.633, -0.011, 0.2634, 0.485], # left back 1 x + 10cm
            [2.31092, -0.24056, 0.99504, -0.21117, 0.12188, 0.41125], # left back 2
            [-2.61095, -0.60861, -0.85194, -0.22472, 0.12920, 0.58045], # left back 3
            [2.95702, -0.59576,  0.03858, -0.04517,  0.17951, 0.49167], # left back 4
            [-2.81997, -0.49272, -0.66790, -0.08580, 0.08629, 0.67964], # left back 5
            [-2.70836, -0.36149, -0.79123, -0.00340, -0.01288, 0.59120] # cleft back
        ]
        # Gripper angle in radians
        tg_gripper_angs = [-np.pi/2 + 0.001] * len(top_cam_cposes)
        conti_move_idxs = [i for i in range(16) if i not in [6, 15]]
        
        linear_idx = [8, 9]
        test_pt = [0, 6, 7, 14]
        self.add_experiment(experiment_name=experiment_name, poses=top_cam_cposes, 
                            gripper_angs=tg_gripper_angs, linear_idx=linear_idx, 
                            conti_move_idxs=conti_move_idxs, test_pt=test_pt)

    def shelf_4cl_sa(self):
        experiment_name = "shelf_4cl_sa"
        # roll pitch yaw x y z in meter
        top_cam_cposes = [
            [3.008, -0.668, 0.549, -0.16574, -0.268, 0.5065], # right back 0 /
            [2.71862, -0.69969, 0.53677, 0.03473, -0.09294, 0.71095], # right back 1
            [-2.23329, -0.26615, -1.04705, -0.19794, -0.12522, 0.42122], #right back 2
            [-2.51890, -0.54126, -0.08807, -0.00074, -0.49342, 0.42817], # right back corner
            [-3.008, -0.668, -0.549, -0.16574, 0.268, 0.5065], # left back 0 /
            [-3.008, -0.5914, -0.633, -0.111, 0.2634, 0.485], # left back 1
            [-3.008, -0.5914, -0.633, -0.061, 0.2634, 0.485], # left back 1 x + 5cm
            [-3.008, -0.5914, -0.633, -0.011, 0.2634, 0.485], # left back 1 x + 10cm
            [2.31092, -0.24056, 0.99504, -0.21117, 0.12188, 0.41125], # left back 2 
            [2.39645, -0.54843, 0.08445, -0.00101, 0.49278, 0.42958], # left back corner

            [2.49654, -0.33109,  0.33812,  0.20139,  0.36231,  0.66868], # left side 0
            [2.45996, -0.30908,  0.27832,  0.14336,  0.37009,  0.61117], # left side 1
            [2.28403, 0.05687, 0.21994, 0.40305, 0.46293, 0.37193], # left side 2

            
            [-1.8205, 0.179, -0.2795, 0.256, -0.5694, 0.157], # right side 0
            [-2.14235, -0.03702, -0.10883,  0.29092, -0.60736,  0.25515], # right side 1
            [2.65787, -0.34722,  0.92235,  0.34916, -0.09080,  0.65794] # right side 2
        ]
        # Gripper angle in radians
        tg_gripper_angs = [-np.pi/2 + 0.001] * len(top_cam_cposes)
        conti_move_idxs = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 13, 14]
        linear_idx = [5, 6]
        test_pt = [0, 4, 10, 15]
        self.add_experiment(experiment_name=experiment_name, poses=top_cam_cposes, 
                            gripper_angs=tg_gripper_angs, linear_idx=linear_idx, 
                            conti_move_idxs=conti_move_idxs, test_pt=test_pt)
        

    def shelf_div_sa(self):
        experiment_name = "shelf_div_sa"
        # roll pitch yaw x y z in meter
        top_cam_cposes = [
            [3.008, -0.668, 0.549, -0.16574, -0.268, 0.5065], # right back 0
            [2.66052, -0.63052,  0.54405, -0.04024, -0.12987, 0.65001], # right back 1
            [-2.31092, -0.24056, -0.99504, -0.21117, -0.12188, 0.41125], #right back 2
            [-2.51890, -0.54126, -0.08807, -0.00074, -0.49342, 0.42817], # right back corner //
            [-2.76276, -0.63924, -0.79016, -0.00504,  0.01017,  0.64309], # left back 0
            [-3.008, -0.5914, -0.633, -0.111, 0.2634, 0.485], # left back 1
            [-3.008, -0.5914, -0.633, -0.061, 0.2634, 0.485], # left back 1 x + 5cm
            [-3.008, -0.5914, -0.633, -0.011, 0.2634, 0.485], # left back 1 x + 10cm
            [-2.71813, -0.86767, -0.69623, -0.06844, 0.09014, 0.69038], # left back 2 //
            [2.39645, -0.54843, 0.08445, -0.00101, 0.49278, 0.42958], # left back corner
            
            [-2.87854, -0.62008, 0.39347, 0.15808, -0.43670, 0.48861], # right cside
            [2.90352, -0.58821, 0.90963, 0.18867, -0.25014, 0.53657], # right side 0
            [-2.89197, -0.22727, 0.26152, 0.38655, -0.20461, 0.65446], # right side 1

            [2.41919, -0.40692, 0.26378, 0.13205, 0.42530, 0.46276], # left cside
            [1.721, 0.179, 0.2195, 0.326, 0.5645, 0.1535], # left side 0
            [2.49879, -0.33148,  0.33862,  0.29275,  0.36147,  0.66898] # left side 1
        ]
        # Gripper angle in radians
        tg_gripper_angs = [-np.pi/2 + 0.001] * len(top_cam_cposes)
        conti_move_idxs = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 13, 14]
        test_pt = [1, 4, 13, 14]
        linear_idx = [5, 6]

        self.add_experiment(experiment_name=experiment_name, poses=top_cam_cposes, 
                            gripper_angs=tg_gripper_angs, linear_idx=linear_idx, 
                            conti_move_idxs=conti_move_idxs, test_pt=test_pt)
        
    def scale_sa(self):
        # A combination of poses from shelf divangs and 4 clusters
        experiment_name = "scale_exp"
        # roll pitch yaw x y z in meter
        top_cam_cposes = [
            [3.008, -0.668, 0.549, -0.16574, -0.268, 0.5065], # right back 0 /
            [2.71862, -0.69969, 0.53677, 0.03473, -0.09294, 0.71095], # right back 1
            [-2.23329, -0.26615, -1.04705, -0.19794, -0.12522, 0.42122], #right back 2
            [-2.51890, -0.54126, -0.08807, -0.00074, -0.49342, 0.42817], # right back corner
            [-2.76276, -0.63924, -0.79016, -0.00504,  0.01017,  0.64309], # left back 0
            [-3.008, -0.5914, -0.633, -0.111, 0.2634, 0.485], # left back 1
            [-3.008, -0.5914, -0.633, -0.061, 0.2634, 0.485], # left back 1 x + 5cm
            [-3.008, -0.5914, -0.633, -0.011, 0.2634, 0.485], # left back 1 x + 10cm
            [-2.71813, -0.86767, -0.69623, -0.06844, 0.09014, 0.69038], # left back 2
            [2.39645, -0.54843, 0.08445, -0.00101, 0.49278, 0.42958], # left back corner

            [2.49654, -0.33109,  0.33812,  0.20139,  0.36231,  0.66868], # left side 0
            [2.45996, -0.30908,  0.27832,  0.14336,  0.37009,  0.61117], # left side 1
            [1.721, 0.179, 0.2195, 0.326, 0.5645, 0.1535], # left side 2

            
            [-1.8205, 0.179, -0.2795, 0.256, -0.5694, 0.157], # right side 0
            [-2.14235, -0.03702, -0.10883,  0.29092, -0.60736,  0.25515], # right side 1
            [2.90352, -0.58821, 0.90963, 0.18867, -0.25014, 0.53657] # right side 2
        ]
        # Gripper angle in radians
        tg_gripper_angs = [-np.pi/2 + 0.001] * len(top_cam_cposes)
        conti_move_idxs = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 13, 14]
        linear_idx = [5, 6]
        test_pt = [0, 4, 12, 15]
        self.add_experiment(experiment_name=experiment_name, poses=top_cam_cposes, 
                            gripper_angs=tg_gripper_angs, linear_idx=linear_idx, 
                            conti_move_idxs=conti_move_idxs, test_pt=test_pt)
        