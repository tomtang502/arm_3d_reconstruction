import pycolmap
import pathlib, os, torch
import numpy as np

from configs.experiments_data_config import ArmDustrExpData
exp_config = ArmDustrExpData()
exp_name = "2bs2sslb3_sa_apriltag" # "4bs3sslb3_sa_apriltag"
output_path = pathlib.Path(exp_config.get_ptc_output_path(exp_name, exp_type=1))
print(output_path)
image_dir = pathlib.Path(os.path.join(exp_config.train_imgs_folder_pth, exp_name))

output_path.mkdir()
mvs_path = output_path / "mvs"
database_path = output_path / "database.db"

pycolmap.extract_features(database_path, image_dir)#, sift_options={"max_num_features": 512})
#pycolmap.extract_features(database_path, image_dir)
pycolmap.match_exhaustive(database_path)
maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
maps[0].write(output_path)
# # dense reconstruction
# pycolmap.undistort_images(mvs_path, output_path, image_dir)

# pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)



# # Step 1: Feature Extraction
# pycolmap.extract_features(database_path, image_dir)

# # Step 2: Feature Matching
# pycolmap.match_exhaustive(database_path)

# # Step 3: Incremental Mapping (Sparse Reconstruction)
# maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)

# Assuming you want to work with the first reconstruction
reconstruction = maps[0]