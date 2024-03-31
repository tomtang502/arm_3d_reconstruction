import subprocess, os
import shutil


# Define the paths to your conda environments' Python interpreters
comap_env_path = '/home/tomtang/anaconda3/envs/colmap/bin/python'
dust3r_env_path = '/home/tomtang/anaconda3/envs/dust3r/bin/python'

# Define the paths to your scripts
runcolmap_path = 'colmap_run.py'
rundust3r_path = 'dust3r_run.py'

# # Define the variable to pass
# exp_name = "8obj_backonly"
# num_imgs = str(11)

exp_name_list = ['8obj_divangs',
    '8obj_4cluster', 
    '8obj_backonly', 

    '7obj_divangs',
    '7obj_4cluster', 
    '7obj_backonly', 
    
    'shelf_divangs',
    'shelf_4cluster',
    'shelf_backonly']
"""
    '8obj_divangs' : 'diverse_ori_sa',
    '8obj_4cluster': 'fourcluster_ori_sa', 
    '8obj_backonly': 'backonly_ori_sa', 

    '7obj_divangs' : 'diverse_ori_sa',
    '7obj_4cluster': 'fourcluster_ori_sa', 
    '7obj_backonly': 'backonly_ori_sa', 
    
    # center of april tag about 0.15 m from base center
    'apriltag_divangs' : 'diverse_ori_sa',
    'apriltag_4cluster': 'fourcluster_ori_sa',
    'apriltag_backonly': 'backonly_ori_sa',
    
    'shelf_divangs' : 'shelf_div_sa',
    'shelf_4cluster': 'shelf_4cl_sa',
    'shelf_backonly': 'backonly_ori_sa',

"""
for exp_name in exp_name_list:
    for i in range(10, 25):
        model_path = f'output/colmap_saved_output/{exp_name}'
        output_path = os.path.join("output/colmap_saved_output", f'{exp_name}_{i}.pth')
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        if not os.path.exists(output_path):
            subprocess.run([comap_env_path, runcolmap_path, exp_name, str(i)])
        else:
            print(output_path, "already done")
    

# Assume script2.py is also modified to accept command-line arguments
# Run script2.py in env2 with exp_name as an argument
#subprocess.run([dust3r_env_path, rundust3r_path, exp_name, num_imgs])
