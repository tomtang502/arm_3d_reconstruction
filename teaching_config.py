from robot_descriptions import z1_description
class TeachingConfig():
    def __init__(self):
        self.dt = 0.002 # movement duration for each integration step
        self.t = 3 # time duration for to move from point to point
        self.end_effector_name = "link06"
        self.urdfloc = z1_description.URDF_PATH
        # joint_min, joint_max, joint_angle_shape can be changed in motion_planning.py
        self.const_vel = 0.10 # m/s
