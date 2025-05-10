import argparse
import itertools
import pprint

from loguru import logger as loguru_logger
import torch

from pipeline.enums import EnvironmentType
from pipeline.Modules.match_finders import AdaMatcherMatchFinder, FeatureDetector, FeatureDetectorMatchFinder, LoFTRMatchFinder
from AdaMatcher.src.utils.metrics import compute_symmetrical_epipolar_errors
from AdaMatcher.src.utils.plotting import make_matching_figures
from AdaMatcher.src.config.default import get_cfg_defaults
from AdaMatcher.src.lightning.data import MultiSceneDataModule

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_cfg_path', type=str, help='data config path')
    parser.add_argument(
        '--model_type',
        type=str,
        default=None,
        choices=['sift', 'adamatcher', 'loftr'],
        help='if set, the matching results will be dump to dump_dir',
    )

    parser.add_argument('--ckpt_path', type=str, required=False, help='Path to the checkpoint')
    parser.add_argument('--index', type=int, required=True, help='Image index')
    parser.add_argument('--figure_path', type=str, required=True, help='PAth to save resulting figure')

    return parser.parse_args()

if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    pprint.pprint(vars(args))

    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    config.merge_from_file(args.data_cfg_path)

    loguru_logger.info(f'Args and config initialized!')

    # lightning data
    args.num_workers = 1
    args.batch_size = 1

    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info(f'DataModule initialized!')

    data_module.setup(stage='test')
    for batch in itertools.islice(data_module.test_dataloader(), args.index, None):
        break

    if args.model_type == "sift":
        matcher = FeatureDetectorMatchFinder(FeatureDetector.SIFT)
    elif args.model_type == "loftr":
        matcher = LoFTRMatchFinder(EnvironmentType.Outdoor, args.ckpt_path)
    elif args.model_type == "adamatcher":
        matcher = AdaMatcherMatchFinder(args.ckpt_path)
    else:
        raise ValueError(f"Unknown matcher: {args.model_type}")
    
    loguru_logger.info(f"img0: {batch['image0'].shape}, img1: {batch['image1'].shape}")

    img0 = batch["image0"].permute(0, 2, 3, 1).cpu().numpy() * 255
    img1 = batch["image1"].permute(0, 2, 3, 1).cpu().numpy() * 255
    loguru_logger.info(f"img0: {img0.shape}, img1: {img1.shape}")
    k1, k2 = matcher.find_matches(img0[0], img1[0])
    batch["mkpts0_f"] = torch.from_numpy(k1).cpu().float()#.unsqueeze(0)
    batch["mkpts1_f"] = torch.from_numpy(k2).cpu().float()#.unsqueeze(0)
    batch["m_bids"] = torch.zeros(k1.shape[0], dtype=torch.long)

    compute_symmetrical_epipolar_errors(
            batch
    ) 

    fig = make_matching_figures(batch, config)
    fig["evaluation"][0].savefig(args.figure_path)
