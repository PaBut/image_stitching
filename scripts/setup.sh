#!/bin/bash

# @Author: Pavlo Butenko

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../"

cd $PROJECT_DIR

mkdir -p "./models/UDIS2/Composition/model/"
mkdir -p "./models/UDIS2/Warp/model/"
mkdir -p "./models/LoFTR/weights/"
mkdir -p "./AdaMatcher/weights/"
mkdir -p "./AdaMatcher/OUTPUT/"

python3 -m gdown "https://drive.google.com/uc?id=1OaG0ayEwRPhKVV_OwQwvwHDFHC26iv30" -O "./models/UDIS2/Composition/model/epoch050_model.pth" # UDIS++ Composition weights
python3 -m gdown "https://drive.google.com/uc?id=1GBwB0y3tUUsOYHErSqxDxoC_Om3BJUEt" -O "./models/UDIS2/Warp/model/epoch100_model.pth"  # UDIS++ Warp weights
python3 -m gdown "https://drive.google.com/uc?id=1M-VD35-qdB5Iw-AtbDBCKC7hPolFW9UY" -O "./models/LoFTR/weights/outdoor_ds.ckpt"  # LoFTR weights
python3 -m gdown "https://drive.google.com/uc?id=1ploO_h0eg7G6WW1xfloDgHHWniO_2Nz7" -O "./AdaMatcher/weights/adamatcher.ckpt"  # AdaMatcher weights