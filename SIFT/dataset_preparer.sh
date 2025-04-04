#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <scene_list.txt> <input_directory> <output_directory>"
    exit 1
fi

scene_list=$1
input_directory=$2
output_directory=$3

while IFS= read -r scene_name; do
    if [ -n "$scene_name" ]; then
        mkdir -p "$output_directory/$scene_name"
        echo "Processing scene: $scene_name"
        colmap model_converter \
            --input_path outputs/$scene_name/gim_dkm/sparse \
            --output_path outputs/$scene_name/gim_dkm/sparse/txt \
            --output_type TXT
        cp "$input_directory/$scene_name/gim_dkm/dense/images" "$output_directory/$scene_name/images" -r
        cp "$input_directory/$scene_name/gim_dkm/dense/stereo/depth_maps" "$output_directory/$scene_name/depth_maps" -r
        python3 npz_builder.py --output "$output_directory/$scene_name.npz" \
                                --cameras_path "$input_directory/$scene_name/gim_dkm/sparse/txt/cameras.txt" \
                                --images_path "$input_directory/$scene_name/gim_dkm/sparse/txt/images.txt"
    fi
done < "$scene_list"