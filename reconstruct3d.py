import subprocess
# Define the paths to your conda environments' Python interpreters
comap_env_path = '/home/tomtang/anaconda3/envs/colmap/bin/python'
dust3r_env_path = '/home/tomtang/anaconda3/envs/dust3r/bin/python'

# Define the paths to your scripts
runcolmap_path = 'colmap_run.py'
rundust3r_path = 'dust3r_run.py'

# Define the variable to pass
exp_name = "7obj_4cluster"
num_imgs = str(18)

# Run script1.py in env1 with exp_name as an argument
subprocess.run([comap_env_path, runcolmap_path, exp_name, num_imgs])

# Assume script2.py is also modified to accept command-line arguments
# Run script2.py in env2 with exp_name as an argument
subprocess.run([dust3r_env_path, rundust3r_path, exp_name, num_imgs])
