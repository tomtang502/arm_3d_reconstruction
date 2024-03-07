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

class ExperimentConfig():
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
    
    def add_experiment(self, poses, grippper_ang, conti_move_idxs, experiment_name, 
                       image_format='jpg'):
        self.exps[]

    @staticmethod
    def twoback_twoside_leftback3linearx():

        