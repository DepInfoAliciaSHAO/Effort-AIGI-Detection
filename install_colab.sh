#!/bin/bash

PYTHON=/usr/bin/python3.7

$PYTHON -m pip install --upgrade pip setuptools

$PYTHON -m pip install numpy==1.21.5
$PYTHON -m pip install pandas==1.4.2
$PYTHON -m pip install Pillow==9.0.1
$PYTHON -m pip install dlib==19.24.0
$PYTHON -m pip install imageio==2.9.0
$PYTHON -m pip install imgaug==0.4.0
$PYTHON -m pip install tqdm==4.61.0
$PYTHON -m pip install scipy==1.7.3
$PYTHON -m pip install seaborn==0.11.2
$PYTHON -m pip install pyyaml==6.0
$PYTHON -m pip install imutils==0.5.4
$PYTHON -m pip install opencv-python==4.6.0.66
$PYTHON -m pip install scikit-image==0.19.2
$PYTHON -m pip install scikit-learn==1.0.2
$PYTHON -m pip install albumentations==1.1.0
$PYTHON -m pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
$PYTHON -m pip install efficientnet-pytorch==0.7.1
$PYTHON -m pip install timm==0.6.12
$PYTHON -m pip install segmentation-models-pytorch==0.3.2
$PYTHON -m pip install torchtoolbox==0.1.8.2
$PYTHON -m pip install tensorboard==2.10.1
$PYTHON -m pip install setuptools==59.5.0
$PYTHON -m pip install loralib
$PYTHON -m pip install einops
$PYTHON -m pip install transformers
$PYTHON -m pip install filterpy
$PYTHON -m pip install simplejson
$PYTHON -m pip install kornia
$PYTHON -m pip install fvcore
$PYTHON -m pip install git+https://github.com/openai/CLIP.git