#!/bin/bash

emb_dir=/home/g21tka10/test/

CUDA_VISIBLE_DEVICES=0 python -u eval.py \
    --pretrained_model_path $1 \
	--data_dir $2 \
	--vocab_dir $2 \
	--glove_dir $emb_dir