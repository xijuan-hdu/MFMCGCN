 #!/bin/bash
# training command for different datasets.
source_dir=../dataset
emb_dir=/home/g21tka10/test
save_dir=saved_models

exp_setting=train

exp_dataset=Biaffine/glove/Laptops
exp_path=$save_dir/Laptops/$exp_setting
if [ ! -d "$exp_path" ]; then
  mkdir -p "$exp_path"
fi

CUDA_VISIBLE_DEVICES=0 python -u train.py \
	--data_dir $source_dir/$exp_dataset \
	--vocab_dir $source_dir/$exp_dataset \
	--glove_dir $emb_dir \
	--input_dropout 0.7 \
	--model "rgat" \
	--batch_size 32 \
	--save_dir $exp_path \
	--hidden_dim 20 \
	--rnn_hidden 20 \
	--head_num_GCN 2 \
	--seed 7 \
	--pooling "avg" \
	--output_merge "gate" \
	--num_layers 2 \
	--attn_heads 5 \
	--num_epoch 100 \
	--shuffle \
	--layer_dropout 0.5 \
	--rnn_dropout 0.5\
  --rnn_layers 2 \
  --lr 0.01 \
  --eps 1e-6 \
  --l2reg 1e-5 \
  --optim "adamax" \
	2>&1 | tee $exp_path/training.log