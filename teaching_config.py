from robot_descriptions import z1_description
class TeachingConfig():
    def __init__(self, rel_path=""):
        self.dt = 0.002 # movement duration for each integration step
        self.tmax = 20
        self.end_effector_name = "gripperStator"
        if rel_path.endswith('/'):
            self.urdf_loc = rel_path + "utils/z1_description/z1_gripper.urdf"
        else:
            self.urdf_loc = rel_path + "/utils/z1_description/z1_gripper.urdf"
        # joint_min, joint_max, joint_angle_shape can be changed in motion_planning.py
        self.const_vel = 0.10 # m/s
        self.initq = [0, -0.001, 0.006, -0.031, -0.079, -0.002, 0.001, 0, 0]
        
        
        """
        Absolute initial position of the gripper loading plane (the J6 end plane): 
            [0,049, 0, 0.1605]
        The position of the center of the Unitree_gripper relative to the loading plane: 
            [0.0382, 0, 0].
        """