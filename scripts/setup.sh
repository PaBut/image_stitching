SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../"

cd $PROJECT_DIR

gdown "https://drive.google.com/uc?id=1OaG0ayEwRPhKVV_OwQwvwHDFHC26iv30" -O "./models/UDIS2/Composition/model/" # UDIS++ Composition weights
gdown "https://drive.google.com/uc?id=1GBwB0y3tUUsOYHErSqxDxoC_Om3BJUEt" -O "./models/UDIS2/Warp/model/"  # UDIS++ Warp weights
gdown "https://drive.google.com/uc?id=1M-VD35-qdB5Iw-AtbDBCKC7hPolFW9UY" -O "./models/LoFTR/weights/"  # LoFTR weights
gdown "https://drive.google.com/uc?id=1ploO_h0eg7G6WW1xfloDgHHWniO_2Nz7" -O "./AdaMatcher/weights/"  # AdaMatcher weights