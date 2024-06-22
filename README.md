# Joint Calibration and Representation (JCR) 

This  repository is the official implementation of the paper [Unifying Scene Representation and Hand-Eye Calibration with 3D Foundation Models](https://pip.pypa.io/en/stable/). It contains a set of experiments we ran along with the implementation of method itself (JCR.py). 


## Installation

It's recommended to use a package manager like [conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) to create a package environment to install and manage required dependencies, and the following installation guide assume a conda base environment is already initiated.

If unzip is not installed via apt, use apt to install unzip before proceed.
```bash
git clone -recursive https://github.com/tomtang502/arm_3d_reconstruction.git
cd arm_3d_reconstruction

conda create -n jcr python=3.11 cmake=3.14.0
conda activate jcr
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

sudo chmod +x installation.sh
./installation.sh
```

## Usage

```python
Some usage code here
```

[Optional] Set up z1 arm for images taking and operations
The Unitree Z1 Robotics Arm was used for experiments, which comes with z1_controller, z1_ros, and z1_sdk (only z1_controller and z1_sdk are used for taking images and operations). Following their [official documentation](https://dev-z1.unitree.com/) to set up z1 arm for experiments.

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
