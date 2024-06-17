#!/bin/bash
CUDA_VISIBLE_DEVICES=5 \
python tools/test.py \
    --config experiments/InstaOrder/midas_pretrained/config.yaml \
    --test_num -1 \
    --load_model '/data/out/InstaOrder_ckpt/model-f6b98070.pt' \
    --pairs "all" \
    --disp_select_method "median" \
#    --disp_select_method "median" \
