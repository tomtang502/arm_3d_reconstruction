
def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Convert euler angles (roll, pitch, yaw) to a rotation matrix.
    """
    R_x = torch.tensor([[1, 0, 0],
                        [0, torch.cos(roll), -torch.sin(roll)],
                        [0, torch.sin(roll), torch.cos(roll)]])
    
    R_y = torch.tensor([[torch.cos(pitch), 0, torch.sin(pitch)],
                        [0, 1, 0],
                        [-torch.sin(pitch), 0, torch.cos(pitch)]])
    
    R_z = torch.tensor([[torch.cos(yaw), -torch.sin(yaw), 0],
                        [torch.sin(yaw), torch.cos(yaw), 0],
                        [0, 0, 1]])
    
    R = torch.matmul(R_z, torch.matmul(R_y, R_x))
    return R



def pose_to_transform(batch):
    """
    Construct a batch of transformation matrices from a batch of roll, pitch, yaw, x, y, z.
    The input batch should have the shape (n, 6), where each row is (roll, pitch, yaw, x, y, z).
    The output will have the shape (n, 4, 4), representing the transformation matrices.
    """
    n = batch.size(0)
    transformation_matrices = torch.zeros((n, 4, 4))
    
    for i in range(n):
        roll, pitch, yaw, x, y, z = batch[i]
        rotation_matrix = rpy_to_rot_matrix_EXC([roll, pitch, yaw])
        transformation_matrix = torch.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = torch.tensor([x, y, z])
        transformation_matrices[i] = transformation_matrix
    
    return transformation_matrices
