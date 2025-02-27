#!/bin/bash
#
# Zhenhao Ge, 2025-02-18


# set configuration file for testing
dataset="Echo2Mix"
experiment="SPMamba-${dataset}"
conf_file="experiments/checkpoint/${experiment}/conf.yml"
[ -f $conf_file ] || (echo $conf_file does not exist! && exit 1)

num_outputs=10
save_output=true
python audio_test.py \
    --conf_file $conf_file \
    --num_outputs $num_outputs \
    --save_output $save_output