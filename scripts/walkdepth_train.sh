#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../"

# conda activate adamatcher
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

data_cfg_path="AdaMatcher/configs/data/walkdepth_trainval.py"
main_cfg_path="AdaMatcher/configs/loftr/outdoor/loftr_ds_walkdepth.py"

n_nodes=1
n_gpus_per_node=1 # 1 4 8
torch_num_workers=8 # 1 4 8
batch_size=1
pin_memory=true
ckpt_path="AdaMatcher/weights/adamatcher.ckpt"
exp_name="AdaMatcher-${TRAIN_IMG_SIZE}-bs$(($n_gpus_per_node * $n_nodes * $batch_size))"

python3 -u ./AdaMatcher/train.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --exp_name=${exp_name} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="cuda" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers} --pin_memory=${pin_memory} \
    --check_val_every_n_epoch=1 \
    --log_every_n_steps=1 \
    --flush_logs_every_n_steps=1 \
    --limit_val_batches=1. \
    --num_sanity_val_steps=200 \
    --benchmark=True \
    --ckpt_path=${ckpt_path} \
    --max_epochs=10 >> ./AdaMatcher/OUTPUT/AdaMatcher.txt