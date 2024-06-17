#!/bin/bash
CUDA_VISIBLE_DEVICES=1 \
python tools/test.py \
    --config experiments/InstaOrder/OrderNet/config.yaml \
    --test_num -1 \
    --load_model "/data/out/InstaOrder_ckpt/InstaOrder_OrderNet.pth.tar" \
    --pairs "all" \
