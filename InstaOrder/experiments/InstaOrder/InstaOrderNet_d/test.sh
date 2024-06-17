#!/bin/bash
CUDA_VISIBLE_DEVICES=5 \
python tools/test.py \
    --config experiments/InstaOrder/InstaOrderNet_d/config.yaml \
    --test_num -1 \
    --pairs "all" \
    --load_model "/data/out/InstaOrder_ckpt/InstaOrder_InstaOrderNet_d.pth.tar" \
    
