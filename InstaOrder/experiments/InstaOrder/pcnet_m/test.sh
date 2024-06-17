#!/bin/bash
CUDA_VISIBLE_DEVICES=5 \
python tools/test.py \
    --config experiments/InstaOrder/pcnet_m/config.yaml \
    --test_num -1 \
    --load_model "/data/out/InstaOrder_ckpt/InstaOrder_pcnet_m.pth.tar" \
    --pairs "all" \
