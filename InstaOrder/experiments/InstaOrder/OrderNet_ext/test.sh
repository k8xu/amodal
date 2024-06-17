#!/bin/bash
CUDA_VISIBLE_DEVICES=3 \
python tools/test.py \
    --config experiments/InstaOrder/OrderNet_ext/config.yaml \
    --test_num -1 \
    --pairs "all" \
    --load_model "/data/out/InstaOrder_ckpt/InstaOrder_OrderNet_ext.pth.tar" \
