#!/bin/bash
CUDA_VISIBLE_DEVICES=0 \
python tools/test.py \
    --config experiments/KINS/pcnet_m/config.yaml \
    --order_th 0.1 \
    --amodal_th 0.2 \
    --test_num -1 \
    --load_model "/data/out/InstaOrder_ckpt/KINS_pcnet_m.pth.tar" \
    --pairs "nbor" \
