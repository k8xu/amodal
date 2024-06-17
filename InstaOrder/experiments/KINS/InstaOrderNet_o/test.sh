#!/bin/bash
CUDA_VISIBLE_DEVICES=6 \
python tools/test.py \
    --config experiments/KINS/InstaOrderNet_o/config.yaml \
    --test_num -1 \
    --load_model "/data/out/InstaOrder_ckpt/KINS_InstaOrderNet_o.pth.tar" \
    --pairs "all" \
