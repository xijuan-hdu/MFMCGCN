#!/usr/bin/bash

source_dir=../dataset
save_dir=saved_models

exp_setting=train
exp_dataset=Biaffine/glove/MAMS



exp_path=$save_dir/MAMS/$exp_setting
if [ ! -d "$exp_path" ]; then
  mkdir -p "$exp_path"
fi

CUDA_VISIBLE_DEVICES=0 python3 -u bert_train.py \
	--lr 2e-5 \
	--bert_lr 2e-5 \
	--input_dropout 0.1 \
	--att_dropout 0 \
	--num_layer 2 \
	--bert_out_dim 100 \
	--attn_heads 2 \
	--hidden_dim 384 \
	--rnn_hidden 384 \
	--rnn_dropout 0.2 \
	--pos_dim 0 \
	--post_dim 0 \
	--rnn_layers 1 \
	--layer_dropout 0.5 \
	--dep_dim 80 \
	--max_len 90 \
	--seed 28 \
	--data_dir $source_dir/$exp_dataset \
	--vocab_dir $source_dir/$exp_dataset \
	--save_dir $exp_path \
	--model "RGAT" \
	--output_merge "gate" \
	--reset_pool \
	--batch_size 16 \
	--num_epoch 10 2>&1 | tee $exp_path/training.log