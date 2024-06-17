#!/bin/bash

# For LaMa inpainter
export TORCH_HOME=$(pwd)/lama && export PYTHONPATH=$(pwd)/lama

# Generate amodal completion
python main.py \
--input_dir dataset/val2014 \
--img_filenames_txt ./img_filenames.txt \
--mc_timestep 35 \
--output_dir ./output
