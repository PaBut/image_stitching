import argparse
import pprint

import pytorch_lightning as pl
from loguru import logger as loguru_logger
import torch

from pipeline.Modules.tools.AdaMatcherUtils.config.default import get_cfg_defaults
from training.src.lightning.data import MultiSceneDataModule
from training.src.lightning.lightning_tester import PL_Tester
from training.src.utils.profiler import build_profiler

from pytorch_lightning.strategies import SingleDeviceStrategy


def parse_args():
    # init a custom parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_cfg_path', type=str, help='data config path')
    parser.add_argument(
        '--dump_dir',
        type=str,
        default=None,
        help='if set, the matching results will be dump to dump_dir',
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default=None,
        choices=['sift', 'adamatcher', 'loftr'],
        help='if set, the matching results will be dump to dump_dir',
    )
    parser.add_argument(
        '--profiler_name',
        type=str,
        default=None,
        help='options: [inference, pytorch], or leave it unset',
    )
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='batch_size per gpu')
    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--ckpt_path', type=str, required=False, help='Path to the checkpoint')

    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs per node')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--accelerator', type=str, default='cuda', help='Type of accelerator (e.g., ddp, dp, etc.)')

    parser.add_argument('--benchmark', action='store_true', help='Enable benchmark mode')

    return parser.parse_args()


if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    pprint.pprint(vars(args))

    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    config.merge_from_file(args.data_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)  # reproducibility

    loguru_logger.info(f'Args and config initialized!')

    # lightning module
    profiler = build_profiler(args.profiler_name)
    model = PL_Tester(
        config,
        matcher_type=args.model_type,
        pretrained_ckpt=args.ckpt_path,
        profiler=profiler,
        dump_dir=args.dump_dir,
    )
    loguru_logger.info(f'AdaMatcher-lightning initialized!')

    # lightning data
    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info(f'DataModule initialized!')

    test_params = {
        "accelerator": args.accelerator,
        "devices": 1,  # Maps 'gpus' to 'devices'
        "num_nodes": args.num_nodes,
        "benchmark": args.benchmark,
    }

    # lightning trainer
    trainer = pl.Trainer(strategy=SingleDeviceStrategy(
                            device=torch.device("cuda:0"),
                        ),
                        logger=True,
                        **test_params)

    loguru_logger.info(f'Start testing!')
    trainer.test(model, datamodule=data_module, verbose=False)
