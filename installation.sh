#!/bin/bash
echo "Install dependencies"
pip install -r requirements.txt --default-timeout=100

echo "Loading checkpoint for dust3r"
# download dust3r checkpoint
FILE_ID="1vka5WSNVJN3Apybxha7jd0YZYeVZDmnR"
FILE_NAME="dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
gdown --id ${FILE_ID} -O ${FILE_NAME}

download experiment images
IMG_FILE_ID="1oM8QTlVR2fPDhUodEoywYJbGfZyWm6az"
IMG_FILE_NAME="arm_captured_images.zip"
gdown --id ${IMG_FILE_ID} -O ${IMG_FILE_NAME}
unzip ${IMG_FILE_NAME}
rm ${IMG_FILE_NAME}

echo "Done"