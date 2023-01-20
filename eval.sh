#!/bin/bash
python src/run_tagger.py \
    --model_name_or_path $1\
    --data_dir ./data/MultiSpanQA_data \
    --validation_file valid.json \
    --output_dir ./eval_output \
    --overwrite_output_dir \
    --overwrite_cache \
    --do_eval \
    --learning_rate 3e-5 \
    --num_train_epochs 0 \
    --max_seq_length  512 \
    --doc_stride 128 > run.out 2>&1
