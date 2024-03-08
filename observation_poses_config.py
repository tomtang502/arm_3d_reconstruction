import numpy as np
class ObsAngsConfig():
    def __init__(self, poses, grippper_angs, experiment_name, 
                 conti_move_idxs=None, image_format='jpg'):
        self.poses = poses
        self.grippper_angs = grippper_angs
        self.name = experiment_name
        if conti_move_idxs == None:
            self.conti_move_idxs = []
        else:
            self.conti_move_idxs = conti_move_idxs
        self.image_format = image_format

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
        # Adding different experiments
        self.twoback_twoside_leftback3linearx()
        self.twoback_twosidesym_leftback3linearx()
        self.twobacksym_twosidesym_leftback3linearx()
        self.fourback3sym_twosidesym_leftback3linearx()
    
    def get_config(self, name):
        if name in self.exps:
            return self.exps[name]
        else:
            raise Exception(f"The experiment with name {name} is not defined in observation_poses_config.py yet!")

    def add_experiment(self, experiment_name, poses, gripper_angs, 
                       conti_move_idxs=None, image_format='jpg'):
        self.exps[experiment_name] = ObsAngsConfig(poses, gripper_angs, experiment_name, 
                                                   conti_move_idxs, image_format)

    # ----------------- example of adding an experiment -----------------
    # Don't fogot to call this in your python file or in __init__!
    def twoback_twoside_leftback3linearx(self):
        experiment_name = "2back_2side_leftback3linearx"
        """
        top hand cam
        right 3.008 -0.668 0.449 -0.16574 -0.268 0.5065 (from start)
        left -3.008 -0.5914 -0.633 -0.111  0.2634  0.485 (from start)
        x+5cm -3.008 -0.5914 -0.633 -0.061  0.2634  0.485
        x+10cm -3.008 -0.5914 -0.633 -0.011  0.2634  0.485

        r_side -1.8205, 0.179, -0.2795, 0.256, -0.5694, 0.157 (from start)
        l_side 1.721, 0.179, 0.2195, 0.326, 0.5645, 0.1535 (from start)
        """
        # roll pitch yaw x y z in meter
        top_cam_cposes = [
            [3.008, -0.668, 0.549, -0.16574, -0.268, 0.5065], # right back
            [-3.008, -0.5914, -0.633, -0.111, 0.2634, 0.485], # left back
            [-3.008, -0.5914, -0.633, -0.061, 0.2634, 0.485], # left back x + 5cm
            [-3.008, -0.5914, -0.633, -0.011, 0.2634, 0.485], # left back x + 10cm
            [-1.8205, 0.179, -0.2795, 0.256, -0.5694, 0.157],
            [1.721, 0.179, 0.2195, 0.326, 0.5645, 0.1535]
        ]
        # Gripper angle in radians
        tg_gripper_angs = [
            -1.5,
            -1.5,
            -1.5,
            -1.5,
            -np.pi/2 + 0.001,
            -np.pi/2 + 0.001
        ]
        conti_move_idxs = [1, 2]
        self.add_experiment(experiment_name=experiment_name, poses=top_cam_cposes, 
                            gripper_angs=tg_gripper_angs, conti_move_idxs=conti_move_idxs)
    
    def twoback_twosidesym_leftback3linearx(self):
        experiment_name = "2back_2sidesym_leftback3linearx"
        # roll pitch yaw x y z in meter
        top_cam_cposes = [
            [3.008, -0.668, 0.549, -0.16574, -0.268, 0.5065], # right back
            [-3.008, -0.5914, -0.633, -0.111, 0.2634, 0.485], # left back

            [-3.008, -0.5914, -0.633, -0.061, 0.2634, 0.485], # left back x + 5cm
            [-3.008, -0.5914, -0.633, -0.011, 0.2634, 0.485], # left back x + 10cm

            [-1.8205, 0.179, -0.2795, 0.256, -0.5694, 0.157], # left side 0
            [-1.721, 0.179, -0.2195, 0.326, -0.5645, 0.1535], # left side 1
            [1.8205, 0.179, 0.2795, 0.256, 0.5694, 0.157], # right side 0
            [1.721, 0.179, 0.2195, 0.326, 0.5645, 0.1535] # right side 1
        ]
        # Gripper angle in radians
        tg_gripper_angs = [
            -1.5,
            -1.5,
            -1.5,
            -1.5,
            -np.pi/2 + 0.001,
            -np.pi/2 + 0.001,
            -np.pi/2 + 0.001,
            -np.pi/2 + 0.001
        ]
        conti_move_idxs = [1, 2, 4, 6]
        self.add_experiment(experiment_name=experiment_name, poses=top_cam_cposes, 
                            gripper_angs=tg_gripper_angs, conti_move_idxs=conti_move_idxs)
    
    def twobacksym_twosidesym_leftback3linearx(self):
        experiment_name = "2backsym_2sidesym_leftback3linearx"
        # roll pitch yaw x y z in meter
        top_cam_cposes = [
            [3.008, -0.668, 0.549, -0.16574, -0.268, 0.5065], # right back 0
            [3.008, -0.5914, 0.633, -0.111, -0.2634, 0.485], # right back 1
            [-3.008, -0.668, -0.549, -0.16574, 0.268, 0.5065], # left back 0
            [-3.008, -0.5914, -0.633, -0.111, 0.2634, 0.485], # left back 1

            [-3.008, -0.5914, -0.633, -0.061, 0.2634, 0.485], # left back x + 5cm
            [-3.008, -0.5914, -0.633, -0.011, 0.2634, 0.485], # left back x + 10cm

            [-1.8205, 0.179, -0.2795, 0.256, -0.5694, 0.157], # left side 0
            [-1.721, 0.179, -0.2195, 0.326, -0.5645, 0.1535], # left side 1
            [1.721, 0.179, 0.2195, 0.326, 0.5645, 0.1535], # right side 0
            [1.8205, 0.179, 0.2795, 0.256, 0.5694, 0.157] # right side 1
        ]
        # Gripper angle in radians
        tg_gripper_angs = [
            -1.5,
            -1.5,
            -1.5,
            -1.5,
            -1.5,
            -1.5,
            -np.pi/2 + 0.001,
            -np.pi/2 + 0.001,
            -np.pi/2 + 0.001,
            -np.pi/2 + 0.001
        ]
        conti_move_idxs = [0, 2, 3, 4, 6, 8]
        self.add_experiment(experiment_name=experiment_name, poses=top_cam_cposes, 
                            gripper_angs=tg_gripper_angs, conti_move_idxs=conti_move_idxs)
        
    def fourback3sym_twosidesym_leftback3linearx(self):
        experiment_name = "4back3sym_2sidesym_leftback3linearx"
        # roll pitch yaw x y z in meter
        top_cam_cposes = [
            [3.008, -0.668, 0.549, -0.16574, -0.268, 0.5065], # right back 0
            [3.008, -0.5914, 0.633, -0.111, -0.2634, 0.485], # right back 1
            [2.72613, -0.16344, 0.39079, -0.09373, -0.04181, 0.36386], #right back 2
            [2.99985, -0.26425, 0.35898, 0.02892, -0.21166, 0.39514], # right back corner
            [-3.008, -0.668, -0.549, -0.16574, 0.268, 0.5065], # left back 0
            [-3.008, -0.5914, -0.633, -0.111, 0.2634, 0.485], # left back 1
            [-3.008, -0.5914, -0.633, -0.061, 0.2634, 0.485], # left back 1 x + 5cm
            [-3.008, -0.5914, -0.633, -0.011, 0.2634, 0.485], # left back 1 x + 10cm
            [-2.72613, -0.16344, -0.39079, -0.09373, 0.04181, 0.36386], # left back 2
            [-3.05392, -0.26651, -0.38712, 0.05153, 0.21865, 0.39747], # left back corner

            [-1.8205, 0.179, -0.2795, 0.256, -0.5694, 0.157], # left side 0
            [-1.721, 0.179, -0.2195, 0.326, -0.5645, 0.1535], # left side 1
            [1.721, 0.179, 0.2195, 0.326, 0.5645, 0.1535], # right side 0
            [1.8205, 0.179, 0.2795, 0.256, 0.5694, 0.157] # right side 1
        ]
        # Gripper angle in radians
        tg_gripper_angs = [
            -1.5,
            -1.5,
            -np.pi/3.5,
            -np.pi/3,
            -1.5,
            -1.5,
            -1.5,
            -1.5,
            -np.pi/3.5,
            -np.pi/3,
            -np.pi/2 + 0.001,
            -np.pi/2 + 0.001,
            -np.pi/2 + 0.001,
            -np.pi/2 + 0.001
        ]
        conti_move_idxs = [0, 1, 2, 4, 5, 6, 7, 8, 10, 12]

        self.add_experiment(experiment_name=experiment_name, poses=top_cam_cposes, 
                            gripper_angs=tg_gripper_angs, conti_move_idxs=conti_move_idxs)
        #-2.72613 -0.16344 -0.39079 -0.09373  0.04181  0.36386 left back center 60 to 75
        # 2.72613 -0.16344 0.39079 -0.09373  -0.04181  0.36386 right back center 60 to 75
        # -3.05392 -0.26651 -0.38712  0.05153  0.21865  0.39747 left back corner 50 to 70
        # 2.99985 -0.26425  0.35898  0.02892 -0.21166  0.39514 right back corner 50 to 70

    # exp
        # test_angs = [2, 8]
        # top_cam_cposes = [top_cam_cposes[i] for i in test_angs]
        # tg_gripper_angs = [tg_gripper_angs[i] for i in test_angs]
        # conti_move_idxs = None
        