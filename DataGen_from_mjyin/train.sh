#!/bin/bash
DATA_TAG="amazon-book" # [beauty, book, software, ml-1m, industrial]

export CUDA_VISIBLE_DEVICES=2,3
export WANDB_PROJECT="DataGen"

accelerate launch --main_process_port 29777 train.py \
    --dataset_name $DATA_TAG \
    --num_train_epochs 200 \
    --per_device_train_batch_size 2048 \
    --learning_rate 1e-3 \
    --output_dir ./outputs/${DATA_TAG} \
    --seed 42 \
    --report_to wandb \
    --run_name $DATA_TAG \
    --logging_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 10 \
    --bf16 True \
    --lr_scheduler_type cosine_with_min_lr \
    --warmup_ratio 0.05 \
    --lr_scheduler_kwargs '{"min_lr_rate": 0.1}'
