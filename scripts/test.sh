#!/bin/bash

# --- 1. 设置环境变量 ---
# 告诉wandb以离线模式运行，避免交互
export WANDB_MODE=offline


# --- 2. 配置核心路径 (请根据你的实际情况检查并修改) ---

# 本地中文BERT模型路径 (必须与训练时使用的完全一致)
PRETRAINED_BERT_PATH=models/bert-base-chinese

# 你的数据目录 (必须与训练时使用的完全一致)
DATA_DIR=dataset/milcause_bert

# 你的输出目录 (必须与训练时使用的完全一致，因为最佳模型保存在这里)
OUTPUT_DIR=output/bert_base_chinese

# 要加载的最佳模型文件的完整路径
LOAD_PATH=$OUTPUT_DIR/best_model_bert.pt


# --- 3. 检查最佳模型文件是否存在 ---
if [ ! -f "$LOAD_PATH" ]; then
    echo "Error: Best model not found at $LOAD_PATH"
    echo "Please run the training script first to generate the model file."
    exit 1
fi

echo "Found best model: $LOAD_PATH"
echo "Starting final evaluation on the test set..."


# --- 4. 执行测试命令 ---
# 注意：在测试时，我们只需要提供模型和数据路径，以及最重要的 --load_path
# 其他训练相关的参数（如lr, epochs等）都会被忽略。
python train.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --transformer_type bert \
    --model_name_or_path $PRETRAINED_BERT_PATH \
    --load_path $LOAD_PATH \
    --test_batch_size 128 \
    --num_class 2 \
    --num_labels 1