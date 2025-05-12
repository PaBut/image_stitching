# Deep Neural Networks for Image Stitching

This project contains a configurable image stitching pipeline that supports both traditional and deep learning-based methods for feature matching and image composition. The implementation supports SIFT, [LoFTR](https://github.com/zju3dv/LoFTR), and [AdaMatcher](https://github.com/TencentYoutuResearch/AdaMatcher) for feature matching and includes several composition techniques, such as basic image overlay, constant and weighted alpha blending, and [UDIS++](https://github.com/nie-lang/udis2) composition module. A custom dataset, generated from video sequences under adverse environmental conditions (e.g., rain, snow, low-light, lens obstructions), is used for evaluation of selected feature matching and AdaMatcher fine-tuning.

## Setup guide

To set up the required environment, start by creating and activating the Conda environment using the following commands:

```
conda env create -f environment.yml
conda activate image_stitching
```

Model weights for AdaMatcher, LoFTR, and UDIS++ can be downloaded by executing the setup script:

```
sh scripts/setup.sh
```

This step requires the `gdown` utility, which is available within the created Conda environment. 

## COLMAP generated dataset

A custom dataset was generated from sequnce of videos retrieved from [Content-Aware Unsupervised Deep Homography Estimation](https://github.com/JirongZhang/DeepHomography) repository using an improved COLMAP 3D reconstruction pipeline adopted from the [GIM framework](https://github.com/xuelunshen/gim). This reconstruction process outputs camera poses, depth maps, undistorted images and records of 3D points. After completing the reconstruction, the output is processed using the [DatasetPreparation/dataset_preparer.sh](DatasetPreparation/dataset_preparer.sh) script, which generates image pairs with corresponding overlap ratios, and retrieves intrinsic, extrinsic matrices and depth maps for individual images. The data focuses on scenes captured in adverse conditions such as rain, snow, low-light, and partially occluded lenses.

Dataset is present in ```datasets/WalkDepth``` directory. Notably, image pair and ground-truth records are stored in `.npz` files with a corresponding name to a scene. Sample image pairs are available in the `images/` directory and follow the naming convention `<pair_id>_pair_<1|2>`.png. 

## Usage

``` bash
python -m scripts.stitcher \
    --img1_path <path_to_image_1> \
    --img2_path <path_to_image_2> \
    --result_path <output_stitched_image_path> \
    --mfinder <sift|adamatcher|loftr> \
    --composition <overlay|simplealpha|weightedalpha|udis2> \
    [--weights <path_to_model_weights>]
```
*The `--weights` argument is optional and only applicable when using AdaMatcher to switch between pretrained and fine-tuned weights.

Python notebook [notebooks/stitching.ipynb](notebooks/stitching.ipynb) is also present for image stitching execution.

## Experiment Evaluation

Evaluation process was adapted from AdaMatcher network with an addition of homography precision metric, average elapsed time and match quantity. Evaluation is conducted using the code located in the `testing/` directory. The evaluation includes key metrics such as pose error (AUC at 5°, 10°, and 20°), epipolar distance precision (threshold of 1e-4), homography estimation precision (AUC at 3, 5, and 10 pixels), average number of matches, and execution time.

To run the benchmark evaluation for a selected feature matcher, use:

``` bash
sh scripts/walkdepth_test.sh <sift|loftr|adamatcher> <weights_path>
```
*`weights_path` argument is ignored for sift option.

Model weights are stored in the following directories:
- `models/LoFTR/weights/outdoor_ds.ckpt` – LoFTR weights;
- `AdaMatcher/weights/adamatcher.ckpt` – baseline AdaMatcher weights;
- `weights/finetuned.ckpt` – fine-tuned AdaMatcher weights.

To visualize feature matches for a specific pair, highlighting valid and invalid correspondences based on epipolar precision threshold, run::

```bash
sh walkdepth_plot <sift|loftr|adamatcher> <weights_path> <pair_index> <output_path>
```

## AdaMatcher Fine-tuning 

To enhance AdaMatcher’s robustness under challenging visual conditions, a fine-tuning procedure was conducted using the generated dataset. The default training configuration involves a learning rate of 8e-8 and is optimized for a single GPU setup, with a recommended minimum of 40 GB VRAM. 

```bash
sh scripts/walkdepth_train.sh
```

In order to enable compatibility with the generated dataset and support newer CUDA environments, several components of the original AdaMatcher implementation were modified or added:

- [AdaMatcher/src/lightning/lightning_adamatcher.py](AdaMatcher/src/lightning/lightning_adamatcher.py) - updated to a newer `pytorch_lightning` version;
- [AdaMatcher/train.py](AdaMatcher/train.py) - updated to a newer `pytorch_lightning` version;
- [AdaMatcher/src/datasets/megadepth.py](AdaMatcher/src/datasets/megadepth.py) - added optional geometry augmentation;
- [AdaMatcher/src/utils/dataset.py](AdaMatcher/src/utils/dataset.py) - modified to read dataset-specific depth map files;
- [AdaMatcher/src/utils/metrics.py](AdaMatcher/src/utils/dataset.py) - added homography precision metric;
- [AdaMatcher/src/utils/augment.py](AdaMatcher/src/utils/augment.py) - added augmentations simulating challenging conditions, such as noise, blur, rain, snow, low-light and lens obstruction;
- [AdaMatcher/configs/data/walkdepth_test.py](AdaMatcher/configs/data/walkdepth_test.py) - dataset config file for evaluation;
- [AdaMatcher/configs/data/walkdepth_trainval.py](AdaMatcher/configs/data/walkdepth_trainval.py) - dataset config file for training;
- [AdaMatcher/configs/loftr/outdoor/loftr_ds_walkdepth.py](AdaMatcher/configs/loftr/outdoor/loftr_ds_walkdepth.py) - training config file;

## Project structure
- `datasets/WalkDepth/` - Prosoposed dataset;
- `datasets/data_module.py` - DataModule for generated dataset for visualization purposes;
- `images/` - Sample images from the generated dataset for image stitching;
- `DatasetPreparation/` - Codes for handling the outputs from a 3D reconstruction pipeline to create the dataset;
- `scripts/` - Scripts for running the image stitching pipeline, performing experiments and fine-tuning, setting up the solution;
- `pipeline/` - Source code of the configurable image stitching pipeline;
- `models/` - Source codes of LoFTR and UDIS++;
- `AdaMatcher/` - Source code of AdaMatcher model and its fine-tuning process;
- `testing/` - Source code for experimental evaluation of select image stitching methods;
- `weights/finetuned.ckpt` - Fine-tuned AdaMatcher weights;
- `environment.yml` - Specification for an Anaconda environment;
- `requirements.txt` - List of Python dependencies.