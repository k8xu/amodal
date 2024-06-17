#!/bin/bash
CUDA_VISIBLE_DEVICES=4 \
python tools/test.py \
    --config experiments/InstaOrder/InstaOrderNet_od/config.yaml \
    --test_num -1 \
    --load_model "/data/out/InstaOrder_ckpt/InstaOrder_InstaOrderNet_od.pth.tar" \
    --pairs "all" \
