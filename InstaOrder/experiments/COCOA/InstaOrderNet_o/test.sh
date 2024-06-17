#!/bin/bash
CUDA_VISIBLE_DEVICES=0 \
python tools/test.py \
    --config experiments/COCOA/InstaOrderNet_o/config.yaml \
    --test_num -1 \
    --load_model "/data/out/InstaOrder_ckpt/COCOA_InstaOrderNet_o.pth.tar" \
    --pairs "all" \
