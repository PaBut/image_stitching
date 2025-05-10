#!/bin/bash -l

# @Author: Pavlo Butenko

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../"

TRAINING_DIR="${PROJECT_DIR}/AdaMatcher"

# conda activate adamatcher
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH:$TRAINING_DIR

# conda activate adamatcher
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

model_type=$1
pretrained_model_path=$2
index=$3
figure_path=$4

data_cfg_path="AdaMatcher/configs/data/walkdepth_test.py"

CUDA_VISIBLE_DEVICES=0 python3 -u ./testing/test_plot.py \
    ${data_cfg_path} \
    --ckpt_path=${pretrained_model_path} \
    --model_type=${model_type} \
    --index=${index} \
    --figure_path=${figure_path}
