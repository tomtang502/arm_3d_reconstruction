import os, torch
import numpy as np
import matplotlib.pyplot as plt
from configs.experiments_data_config import ArmDustrExpData
from tqdm import tqdm
exp_config = ArmDustrExpData()


def seg_experiment(exp_name, num_imgs, seg_file_suffix='.pth'):
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
    #print(len(seg_mat_L))


    # ---------------------------- #
    # load caliberated point cloud #
    # ---------------------------- #
    dust3r_meta = torch.load(dust3r_caliberated_path)
    pts = dust3r_meta['dense_pt']
    pts_loc = dust3r_meta['pt_loc']
    pts_class = np.zeros((pts.shape[0], 1), dtype=int)
    print(pts.shape, pts_loc.shape)
    print(pts_class.shape)

    for i in tqdm(range(pts.shape[0])):
        row, col, img_idx = pts_loc[i]
        pts_class[i] = seg_mat_L[img_idx][row, col]

    print(np.sum(pts_class), pts_class.shape, "should be from", pts.shape)

    tensors_to_save = {
            'poses': dust3r_meta['poses'],
            'dense_pt': dust3r_meta['dense_pt'],
            'pt_class': pts_class,
            'colors': dust3r_meta['colors'],
            'pt_loc' : dust3r_meta['pt_loc'], # row, col, img_idx
            'eef_poses': dust3r_meta['eef_poses'],
            'T' : dust3r_meta['T'],
            'eef_idx': dust3r_meta['eef_idx'],
            'J' : dust3r_meta['J'],
            'trans_L' : dust3r_meta['trans_L'],
            'rot_L' : dust3r_meta['rot_L']
    }
    saving_path = os.path.join("output/dust3r_segmented_output", f'{exp_name}_{num_imgs}.pth')
    torch.save(tensors_to_save, saving_path)
    print("-"*10)
    print(f"dust3r out saved at {saving_path}")
    print("-"*10)

if __name__ == "__main__":
    """
    Experiment names:
    '8obj_divangs', '8obj_4cluster',
    '7obj_divangs', '7obj_4cluster',
    'shelf_divangs', 'shelf_4cluster'
    """
    exp_name_list = ['5obj_measure']
    for exp_name in exp_name_list:
        for i in range(6, 13, 2): 
            seg_experiment(exp_name, i)
    
