import json, os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.vis.plotly_vis import plot_scene

from ray_diffusion.dataset import CustomDataset
from ray_diffusion.inference.load_model import load_model
from ray_diffusion.inference.predict import predict_cameras

from PIL import Image
import tempfile
import shutil
import os

def get_image_shape_to_bboxes(image_directory):
    files = os.listdir(image_directory)
    image_files = [f for f in files if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]
    if not image_files:
        raise FileNotFoundError("No image files found in the directory")
    
    first_image_path = os.path.join(image_directory, image_files[0])
    with Image.open(first_image_path) as img:
        width, height = img.size
    
    return [[0, 0, width, height] for _ in range(len(image_files))]


def run_raydiffusion(image_dir, model_dir, bboxes):
    device = torch.device("cuda:0")
    model, cfg = load_model(model_dir, device=device)
    dataset = CustomDataset(
        image_dir=image_dir,
        mask_dir='',
        bboxes=bboxes,
        mask_images=False,
    )
    num_frames = dataset.n
    batch = dataset.get_data(ids=np.arange(num_frames))
    images = batch["image"].to(device)
    crop_params = batch["crop_params"].to(device)

    is_regression = cfg.training.regression
    if is_regression:
        # regression
        pred = predict_cameras(
            model=model,
            images=images,
            device=device,
            pred_x0=cfg.model.pred_x0,
            crop_parameters=crop_params,
            use_regression=True,
        )
        predicted_cameras = pred[0]
    else:
        # diffusion
        pred = predict_cameras(
            model=model,
            images=images,
            device=device,
            pred_x0=cfg.model.pred_x0,
            crop_parameters=crop_params,
            additional_timesteps=(70,),  # We found that X0 at T=30 is best.
            rescale_noise="zero",
            use_regression=False,
            max_num_images=None if num_frames <= 8 else 8,  # Auto-batch for N > 8.
            pbar=True,
        )
        predicted_cameras = pred[1][0]

    # Visualize cropped and resized images
    n = (predicted_cameras.R).shape[0]
    T = torch.zeros((n, 4, 4))
    T[:, :3, :3] = (predicted_cameras.R).clone().to('cpu')
    T[:, :3, 3] = (predicted_cameras.T).clone().to('cpu')
    T[:, 3, 3] = 1.0

    return T

if __name__ == "__main__":
    exp_names = [
        '8obj_divangs',
        '7obj_divangs',
        'shelf_divangs'
    ]
    predicted_poses = dict()
    unused_points = [1, 4, 13, 14]
    #[1, 4, 13, 14] b e n o are not used because they are testing points used to test JCR.
    for exp_name in exp_names:
        img_dir = f"arm_captured_images/{exp_name}"
        image_files = sorted(os.listdir(img_dir))
        image_files_used = [image_files[i] for i in range(len(image_files)) if i not in unused_points]
        for num_img in [12, 15]:
            with tempfile.TemporaryDirectory() as temp_dir:
                for file in image_files_used[:num_img]:
                    # Ensure the file exists before copying
                    file_path = os.path.join(img_dir, file)
                    if os.path.exists(file_path):
                        shutil.copy(file_path, temp_dir)
                    else:
                        print(f"File {file_path} does not exist and will not be copied.")
                # temp_files = os.listdir(temp_dir)
                # print("Files in temporary directory:")
                # print(temp_files)
                T = run_raydiffusion(image_dir=temp_dir, model_dir="models/co3d_diffusion", 
                                bboxes=get_image_shape_to_bboxes(temp_dir))
                predicted_poses[f"{exp_name}_{num_img}"] = T
    print(predicted_poses)
    torch.save(predicted_poses, "rdiff_output.pt")
    
