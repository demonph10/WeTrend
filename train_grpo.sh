#!/bin/bash

# GPU配置
export CUDA_VISIBLE_DEVICES=6,7

# 奖励权重配置
export TS_REWARD_ACCURACY_WEIGHT=0.6     # 准确度奖励权重

# 设置模型路径
MODEL_PATH=""     #设置Trender-Base路径
export QWEN_MODEL_PATH=$MODEL_PATH

# 输出和数据集路径
OUTPUT_DIR="./output/trender-pro"
DATASET_PATH="trender_stage2.json"

MASTER_PORT=29519 NPROC_PER_NODE=2 swift rlhf \
    --rlhf_type grpo \
    --model $MODEL_PATH \
    --external_plugins plugin/reward.py \
    --deepspeed zero3 \
    --reward_funcs reward \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset $DATASET_PATH \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir $OUTPUT_DIR \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 4 \
    --temperature 0.9 \
    --log_completions true \
    --seed 42