import pycolmap
import pathlib, os#, torch
import numpy as np
import shutil

from configs.experiments_data_config import ArmDustrExpData
exp_config = ArmDustrExpData()
exp_name = "7obj_4cluster"

import os
import shutil

def copy_images_to_tmp(original_folder, idxs, parent_folder):
    """
    Copy specified images from the original folder to a temporary folder under the specified parent folder.

    Args:
    original_folder: Path to the original folder containing the images.
    image_names: List of image file names to copy.
    parent_folder: Path to the parent folder where the temporary folder should be created.

    Returns:
    Path to the temporary folder where the images are copied.
    """
    # Create a temporary directory under the parent folder
    tmp_folder = os.path.join(parent_folder, "tmp")
    os.makedirs(tmp_folder, exist_ok=True)

    i =0 
    filenames = os.listdir(original_folder)
    filenames.sort()
    # Copy images to the temporary folder
    for filename in filenames:
        if (filename.endswith(('.jpg', '.jpeg', '.png', '.gif')) and
            i not in idxs):
            original_path = os.path.join(original_folder, filename)
            if os.path.isfile(original_path):
                shutil.copy(original_path, tmp_folder)
        i += 1

    return tmp_folder

def delete_tmp_folder(tmp_folder):
    """
    Delete the temporary folder.

    Args:
    tmp_folder: Path to the temporary folder to delete.
    """
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)

# Example usage:

output_path = pathlib.Path(exp_config.get_ptc_output_path(exp_name, exp_type=1))
original_folder = exp_config.get_images_dir(exp_name)
pose_data = exp_config.get_obs_config(exp_name)
tmp_folder = copy_images_to_tmp(original_folder, pose_data.test_pt, "output")
# Copy images to the temporary folder under the parent folder
print("Images copied to temporary folder:", tmp_folder)
print(output_path)
image_dir = pathlib.Path(tmp_folder)

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


# Delete the temporary folder
delete_tmp_folder(tmp_folder)
print("Temporary folder deleted.")