#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../"

# conda activate adamatcher
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

model_type=$1
pretrained_model_path=$2

data_cfg_path="training/configs/data/walkdepth_test_832.py"

dump_dir="dump/loftr_ds_outdoor"
profiler_name="inference"
n_nodes=1  # manually keep this the same with --nodes
n_gpus_per_node=1 # -1
torch_num_workers=8 # 4
batch_size=1  # per gpu

CUDA_VISIBLE_DEVICES=0 python3 -u ./testing/test_lightning.py \
    ${data_cfg_path} \
    --ckpt_path=${pretrained_model_path} \
    --model_type=${model_type} \
    --dump_dir=${dump_dir} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="cuda" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers}\
    --profiler_name=${profiler_name} \
    --benchmark
