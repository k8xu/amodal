#!/bin/bash
CUDA_VISIBLE_DEVICES=6 \
python tools/test.py \
    --config experiments/KINS/OrderNet/config.yaml \
    --order_th 0.1 \
    --test_num -1 \
    --load_model "/data/out/InstaOrder_ckpt/KINS_OrderNet.pth.tar" \
    --pairs "all" \
