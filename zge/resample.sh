#!/bin/bash
#
# resample audio files from 16k to 8k
# 
# Zhenhao Ge, 2025-02-27

# Define source and target directories
SOURCE_DIR="/home/users/zge/data1/datasets/Echo2Mix/16k"
TARGET_DIR="/home/users/zge/data1/datasets/Echo2Mix/8k"

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Find all .wav files and store them in an array
mapfile -t wav_files < <(find "$SOURCE_DIR" -type f -name "*.wav")

# Loop through each .wav file for downsampling
for file in "${wav_files[@]}"; do
    
    # Define output file path in the target directory
    file2="${file//$SOURCE_DIR/$TARGET_DIR}"

    target_dir=$(dirname "$file2")
    mkdir -p "$target_dir"

    # Downsample the .wav file using SoX
    if [[ -f "$file2" ]]; then
        echo "Skipping $file2 already exist."
    else     
        sox "$file" -r 8000 "$file2"
        echo "Downsampled: $file -> $file2"
    fi

done

echo "Processing complete!"