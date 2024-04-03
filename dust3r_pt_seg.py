import os, torch
import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
from configs.experiments_data_config import ArmDustrExpData
exp_config = ArmDustrExpData()



exp_name = '8obj_divangs'
num_imgs = 10
seg_file_suffix = '.pth'

"""
constant
"""
num_test_imgs = 4
num_train_imgs_max = 20

segmented_img_dir = f"output/tiny_sam_output/{exp_name}"
dust3r_caliberated_path = os.path.join("output/dust3r_saved_output", f'{exp_name}_{num_imgs}.pth')
pose_data = exp_config.get_obs_config(exp_name)

# --------------------- #
# load segmented images #
# --------------------- #
seg_files = [os.path.join(segmented_img_dir, file) 
             for file in os.listdir(segmented_img_dir) 
             if file.endswith(seg_file_suffix)]

seg_files.sort()
seg_mat_L = []
for file_path in seg_files[:num_test_imgs+num_train_imgs_max]:
    img_seg_meta = torch.load(file_path)
    seg_mat_L.append(img_seg_meta['pix_cla'])
    #plt.imshow(img_seg_meta['viz'])
    #plt.show()
#seg_f_L = [seg_files[i] for i in range(len(seg_mat_L)) if i not in pose_data.test_pt]
seg_mat_L = [seg_mat_L[i] for i in range(len(seg_mat_L)) if i not in pose_data.test_pt]
print(len(seg_mat_L))


# ---------------------------- #
# load caliberated point cloud #
# ---------------------------- #
dust3r_meta = torch.load(dust3r_caliberated_path)
pts = dust3r_meta['dense_pt']
pts_loc = dust3r_meta['pt_loc']
pts_class = np.zeros((pts.shape[0], 1), dtype=int)
print(pts.shape, pts_loc.shape)
print(pts_class.shape)

for i in range(pts.shape[0]):
    row, col, img_idx = pts_loc[i]
    pts_class[i] = seg_mat_L[img_idx][row, col]

# -------- #
# Plotting #
# -------- #

print(pts_class.sum())

# Create a figure
fig = mlab.figure()
pts_class = pts_class
pts = pts.numpy()
# Plot points with different colors for each class
for c in np.unique(pts_class):
    ix = np.where(pts_class == c)
    color = tuple(list(np.random.rand(3)))
    mlab.points3d(pts[ix, 0], pts[ix, 1], pts[ix, 2], color=color, scale_factor=0.05, scale_mode='none', mode='point')

# Add axes labels
mlab.xlabel('X')
mlab.ylabel('Y')
mlab.zlabel('Z')

# Display the plot
mlab.show()

#row, col, img_idx

