#/bin/bash!
#
# Combine json file pairs to prepare the data lists from two datasets
# (e.g., WSJ0-2Mix and Echo2Mix)
#
# Zhenhao Ge, 2025-03-06


dataset1="WSJ0-2Mix"
dataset2="Echo2Mix"
dataset="WSJ0-Echo-2Mix"
sr=16k

data_dir=data
# out_dir=${data_dir}/${dataset}/${sr}
# mkdir -p ${out_dir}

# looping
categories=("train" "val" "test")
sources=("mix" "s1" "s2")
for cat in "${categories[@]}"; do
    for src in "${sources[@]}"; do
        echo "Procesing category: $cat, source: $src"
        json_infile1=${data_dir}/${dataset1}/${sr}/${cat}/${src}.json
        json_infile2=${data_dir}/${dataset2}/${sr}/${cat}/${src}.json
        out_dir=${data_dir}/${dataset}/${sr}/${cat}
        mkdir -p $out_dir
        json_outfile=${out_dir}/${src}.json
        python zge/combine_jsons.py \
            --json-infile1 $json_infile1 \
            --json-infile2 $json_infile2 \
            --json-outfile $json_outfile
    done
done