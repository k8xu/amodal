#!/bin/bash
CUDA_VISIBLE_DEVICES=3 \
python tools/test.py \
    --config experiments/InstaOrder/InstaDepthNet_d/config.yaml \
    --test_num -1    \
    --pairs "all" \
    --load_model "/data/out/InstaOrder_ckpt/InstaOrder_InstaDepthNet_d.pth.tar"
