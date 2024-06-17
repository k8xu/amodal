#!/bin/bash

# Set up diffusers
cd diffusers
pip install -e ".[torch]"
cd ..

# Download Grounding DINO and SAM model checkpoints
cd Grounded-Segment-Anything
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ..

# Set up InstaOrder
cd InstaOrder
gdown 1_GEmCmofLSkJZnidfp4vsQb2Nqq5aqBU
unzip InstaOrder_ckpt.zip
rm InstaOrder_ckpt.zip
rm -rf __MACOSX
cd ..

# Set up LaMa
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install requirements
pip install -r requirements.txt
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
