#!/bin/bash
# training command for different datasets.
source_dir=../dataset
emb_dir=/home/g21tka10/test
save_dir=saved_models

exp_setting=train

exp_dataset=Biaffine/glove/MAMS
exp_path=$save_dir/MAMS/$exp_setting
if [ ! -d "$exp_path" ]; then
  mkdir -p "$exp_path"
fi


CUDA_VISIBLE_DEVICES=0 python -u train.py \
	--data_dir $source_dir/$exp_dataset \
	--vocab_dir $source_dir/$exp_dataset \
	--glove_dir $emb_dir \
	--model "RGAT" \
	--save_dir $exp_path \
	--seed 29 \
	--batch_size 128 \
	--hidden_dim 50 \
	--rnn_hidden 50 \
	--pooling "avg" \
	--output_merge "gate" \
	--num_layers 4 \
	--attn_heads 5 \
	--num_epoch 100 \
	--shuffle \
	--layer_dropout 0.5 \
  --rnn_dropout 0.3 \
  --rnn_layers 2 \
  --lr 0.01 \
  --is_first True \
  --optim "adamax"
	2>&1 | tee $exp_path/training.log