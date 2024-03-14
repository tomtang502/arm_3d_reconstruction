import pycolmap
import pathlib, os#, torch
import numpy as np

from configs.experiments_data_config import ArmDustrExpData
exp_config = ArmDustrExpData()
exp_name = "8obj_divangs" # "4bs3sslb3_sa_apriltag"
output_path = pathlib.Path(exp_config.get_ptc_output_path(exp_name, exp_type=1))
print(output_path)
image_dir = pathlib.Path(exp_config.get_images_dir(exp_name))

output_path.mkdir()
mvs_path = output_path / "mvs"
database_path = output_path / "database.db"

pycolmap.extract_features(database_path, image_dir)#, sift_options={"max_num_features": 512})
#pycolmap.extract_features(database_path, image_dir)
pycolmap.match_exhaustive(database_path)
maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
maps[0].write(output_path)

pycolmap.undistort_images(mvs_path, output_path, image_dir)
pycolmap.patch_match_stereo(mvs_path)  # requires compilation with CUDA
pycolmap.stereo_fusion(output_path / "dense.ply", mvs_path)
