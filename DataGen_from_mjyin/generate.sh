#!/bin/bash
DATA_TAG="amazon-industrial" # [beauty, book, software, ml-1m, industrial]
CKPT_PATH="outputs/amazon-industrial/checkpoint-5000"

export CUDA_VISIBLE_DEVICES=2,3
export WANDB_PROJECT="DataGen"

mkdir -p data_generated/${DATA_TAG}-gen/
cp data_transformed/${DATA_TAG}/${DATA_TAG}.valid.inter data_generated/${DATA_TAG}-gen/${DATA_TAG}-gen.valid.inter
cp data_transformed/${DATA_TAG}/${DATA_TAG}.test.inter data_generated/${DATA_TAG}-gen/${DATA_TAG}-gen.test.inter

accelerate launch --num_processes 2 --main_process_port 29778 generate.py \
    --dataset_name $DATA_TAG \
    --ckpt_path $CKPT_PATH
