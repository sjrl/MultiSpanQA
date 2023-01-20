#!/bin/bash
python src/run_tagger.py \
    --model_name_or_path bert-base-uncased \
    --data_dir ./data/MultiSpanQA_data \
    --train_file train.json \
    --validation_file valid.json \
    --output_dir ./output \
    --overwrite_output_dir \
    --overwrite_cache \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 4 \
    --eval_accumulation_steps 50 \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --max_seq_length  512 \
    --doc_stride 128 > run.out 2>&1
