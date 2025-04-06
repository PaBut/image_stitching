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
        input_path=""
        
        if [ -e "$input_directory/$scene_name/gim_dkm/sparse/cameras.bin" ]; then
            input_path="$input_directory/$scene_name/gim_dkm/sparse"
        elif [ -e "$input_directory/$scene_name/gim_lightglue/sparse/cameras.bin"]; then
            input_path="$input_directory/$scene_name/gim_lightglue/sparse"
        else
            echo "No sparse model found for scene: $scene_name"
            continue
        fi

        mkdir -p "$output_directory/$scene_name"
        mkdir -p "$input_path/txt"
        echo "Processing scene: $scene_name"
        colmap model_converter \
            --input_path $input_path \
            --output_path $input_path/txt \
            --output_type TXT
        cp "$input_directory/$scene_name/gim_dkm/dense/images" "$output_directory/$scene_name/images" -r
        cp "$input_directory/$scene_name/gim_dkm/dense/stereo/depth_maps" "$output_directory/$scene_name/depth_maps" -r
        python3 npz_builder.py --output "$output_directory/$scene_name.npz" \
                                --cameras_path "$input_path/txt/cameras.txt" \
                                --images_path "$input_path/txt/images.txt" \
                                --points3D_path "$input_path/txt/points3D.txt"
    fi
done < "$scene_list"