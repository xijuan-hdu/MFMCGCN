#!/usr/bin/bash

source_dir=../dataset
save_dir=saved_models

exp_setting=train
exp_dataset=Biaffine/glove/Tweets



exp_path=$save_dir/Tweets/$exp_setting
if [ ! -d "$exp_path" ]; then
  mkdir -p "$exp_path"
fi


	CUDA_VISIBLE_DEVICES=0 python3 -u bert_train.py \
	--lr 2e-5 \
	--bert_lr 2e-6 \
	--input_dropout 0.1 \
	--att_dropout 0.0 \
	--num_layer 2 \
	--bert_out_dim 768 \
	--attn_heads 2 \
	--hidden_dim 100 \
	--rnn_hidden 100 \
	--rnn_dropout 0.2 \
	--pos_dim 30 \
	--post_dim 30 \
	--rnn_layers 1 \
	--layer_dropout 0.5 \
	--dep_dim 100 \
	--max_len 100 \
	--data_dir $source_dir/$exp_dataset \
	--vocab_dir $source_dir/$exp_dataset \
	--save_dir $exp_path \
	--model "RGAT" \
	--seed 55 \
	--output_merge "gate" \
	--reset_pool \
	--num_epoch 15 2>&1 | tee $exp_path/training.log