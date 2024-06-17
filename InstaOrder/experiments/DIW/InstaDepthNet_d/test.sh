#!/bin/bash
CUDA_VISIBLE_DEVICES=2 \
python tools/test_disp_DIW.py \
    --config experiments/DIW/InstaDepthNet_d/config.yaml \
    --test_num -1 \
    --load_model "/data/out/InstaOrder_ckpt/InstaOrder_InstaDepthNet_d.pth.tar" 
