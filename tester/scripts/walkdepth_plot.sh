#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../"

# conda activate adamatcher
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

model_type=$1
pretrained_model_path=$2
index=$3
figure_path=$4

data_cfg_path="tester/configs/walkdepth_test_config.py"

dump_dir="dump/loftr_ds_outdoor"
profiler_name="inference"
n_nodes=1  # manually keep this the same with --nodes
n_gpus_per_node=1 # -1
torch_num_workers=8 # 4
batch_size=1  # per gpu

CUDA_VISIBLE_DEVICES=0 python3 -u ./test_plot.py \
    ${data_cfg_path} \
    --ckpt_path=${pretrained_model_path} \
    --model_type=${model_type} \
    --index=${index} \
    --figure_path=${figure_path}
