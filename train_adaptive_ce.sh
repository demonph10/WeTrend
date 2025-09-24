#!/bin/bash

# GPU配置
export CUDA_VISIBLE_DEVICES=6,7

# 设置模型路径
MODEL_PATH="./Qwen3-8B-base"
export QWEN_MODEL_PATH=$MODEL_PATH

OUTPUT_DIR="./output/trender-base"
DATASET_PATH="trender_stage1.json"

# 使用环境变量启动分布式训练
MASTER_PORT=29517 NPROC_PER_NODE=2 swift sft \
    --model $MODEL_PATH \
    --deepspeed zero3 \
    --loss_type adaptive_ce \
    --external_plugins plugin/adaptive_ce.py \
    --train_type full \
    --dataset $DATASET_PATH \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 2 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 2048 \
    --output_dir $OUTPUT_DIR \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --seed 42 \
    --load_from_cache_file false