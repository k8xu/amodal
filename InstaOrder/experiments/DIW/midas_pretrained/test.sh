#!/bin/bash
CUDA_VISIBLE_DEVICES=0 \
python tools/test_disp_DIW.py \
    --config experiments/DIW/midas_pretrained/config.yaml \
    --test_num -1 \
    --load_model '/data/out/InstaOrder_ckpt/model-f6b98070.pt'
