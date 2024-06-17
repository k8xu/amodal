#!/bin/bash
CUDA_VISIBLE_DEVICES=2 \
python tools/test.py \
    --config experiments/COCOA/pcnet_m/config.yaml \
    --test_num -1 \
    --load_model "/data/out/InstaOrder_ckpt/COCOA_pcnet_m.pth.tar" \
    --pairs "all" \
