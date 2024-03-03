from robot_descriptions import z1_description
class TeachingConfig():
    def __init__(self, rel_path=""):
        self.dt = 0.002 # movement duration for each integration step
        self.t = 3 # time duration for to move from point to point
        self.end_effector_name = "gripperStator"
        if rel_path.endswith('/'):
            self.urdf_loc = rel_path + "utils/z1_description/z1_gripper.urdf"
        else:
            self.urdf_loc = rel_path + "/utils/z1_description/z1_gripper.urdf"
        # joint_min, joint_max, joint_angle_shape can be changed in motion_planning.py
        self.const_vel = 0.10 # m/s
