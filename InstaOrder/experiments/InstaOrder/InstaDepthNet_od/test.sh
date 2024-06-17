#!/bin/bash
CUDA_VISIBLE_DEVICES=2       \
python tools/test.py \
    --config experiments/InstaOrder/InstaDepthNet_od/config.yaml \
    --test_num -1    \
    --pairs "all" \
    --load_model "/data/out/InstaOrder_ckpt/InstaOrder_InstaDepthNet_od.pth.tar" \

