# Paper Citation
```bash
@ARTICLE{zhiJCR_2024,
  author={Zhi, Weiming and Tang, Haozhan and Zhang, Tianyi and Johnson-Roberson, Matthew},
  journal={IEEE Robotics and Automation Letters}, 
  title={Unifying Representation and Calibration With 3D Foundation Models}, 
  year={2024},
}
```

# Joint Calibration and Representation (JCR) 

This  repository is the official implementation of the paper [Unifying Scene Representation and Hand-Eye Calibration with 3D Foundation Models](https://arxiv.org/abs/2404.11683). It contains a set of experiments we ran along with the implementation of method itself (JCR.py). 
![Method Demonstration](/configs/fig1.png)

## Installation

It's recommended to use a package manager like [conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) to create a package environment to install and manage required dependencies, and the following installation guide assume a conda base environment is already initiated.

If unzip is not installed via apt, use apt to install unzip before proceed.
```bash
git clone --recursive https://github.com/tomtang502/arm_3d_reconstruction.git
cd arm_3d_reconstruction

conda create -n jcr python=3.11 cmake=3.14.0
conda activate jcr
# change to the your cuda-toolkit version, and I think pytorch version can be flexible (especially if you need compatibility with other module envs).
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# The following commands also download some experiment images we have taken via robotic arm.
sudo chmod +x installation.sh 
./installation.sh
```

## Usage

To download some sample experiments data we've processed by dust3r and colmap (some of them are processed by tiny-sam for segmentation visualization).
```bash
sudo chmod +x download_data.sh
./download_data.sh
```
[JCR_run.py](JCR_run.py) walk through how to use the JCR method which take in images, and output caliberated poses and point cloud.
```bash
python JCR_run.py
```
[Optional] For segmentation and visualization
The directory [tiny_sam](tiny_sam) contains scripts that requires installation of [TinySam](https://github.com/xinghaochen/TinySAM.git) to segment the input images.

Then, those segmentation masks of the input images (for details look into the script at [tiny_sam_seg.py](tiny_sam/tiny_sam_seg.py)), which can then be used to conduct segmented point cloud via [JCR_pt_seg.py](JCR_pt_seg.py). The result can be used by [precision_measurement.py](precision_measurement.py) to visualize and interact with the scene captured by images of this result.

[Optional] Set up z1 arm for images taking and operations
The Unitree Z1 Robotics Arm was used for experiments, which comes with z1_controller, z1_ros, and z1_sdk (only z1_controller and z1_sdk are used for taking images and operations). Following their [official documentation](https://dev-z1.unitree.com/) to set up z1 arm for experiments, and [generate_images.py](generate_images.py) can be used to take images via z1 arm and a camera mouted on top of it.

[MISC]
[ray_diffusion_comparison](ray_diffusion_comparison) contains code we used to compare with Ray Diffusion, which requires installing [Ray Diffusion](https://github.com/jasonyzhang/RayDiffusion.git) following their official instruction.

[colmap_comparison](colmap_comparison) contains code we used to compare with Colmap, which requires build pycolmap from source, and the instruction can be found [here](https://colmap.github.io/).

## Acknowledgment

We thank Thomas Luo @ CMU for helping us processing our image data.

## Attribution

This repository includes a module licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

Module: Dust3r

Original Authors: Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, Jérôme Revaud

Source: https://github.com/naver/dust3r

License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)


## License

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en)
