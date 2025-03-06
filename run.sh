#!/bin/bash
#
# Zhenhao Ge, 2025-02-18

# set dataset
dataset="WSJ0" # Echo2Mix, WSJ0

# training
conf_dir=configs/spmamba-wsj0.yml
python audio_train.py \
    --conf_dir $conf_dir

# set configuration file for testing
experiment="SPMamba-${dataset}"
conf_file="experiments/checkpoint/${experiment}/conf.yml"
[ -f $conf_file ] || (echo $conf_file does not exist! && exit 1)

# testing
model_path=experiments/checkpoint/SPMamba-WSJ0/ep153.pth
dataset="WSJ0-2Mix"
sample_rate=16000
num_outputs=10
save_output=true
python audio_test.py \
    --conf_file $conf_file \
    --model_path $model_path \
    --dataset $dataset \
    --sample_rate $sample_rate \
    --num_outputs $num_outputs \
    --save_output $save_output