#!/bin/bash

export WANDB_MODE=offline

python train.py \
    --data_dir dataset/milcause_roberta \
    --output_dir output/chinese-roberta-wwm-ext \
    --transformer_type roberta \
    --model_name_or_path models/chinese-roberta-wwm-ext \
    --load_path output/bert-base-chinese/*.pt \
    --test_batch_size 64 \
    --num_labels 1 \
    --num_class 2
