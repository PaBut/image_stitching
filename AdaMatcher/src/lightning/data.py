import math
import os
from collections import abc
from os import path as osp
from pathlib import Path

import pytorch_lightning as pl
from joblib import Parallel, delayed
from loguru import logger
from torch import distributed as dist
from torch.utils.data import (ConcatDataset, DataLoader, Dataset,
                              DistributedSampler, RandomSampler, dataloader)
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from AdaMatcher.src.datasets.megadepth import MegaDepthDataset
from AdaMatcher.src.datasets.sampler import RandomConcatSampler
from AdaMatcher.src.datasets.scannet import ScanNetDataset
from AdaMatcher.src.utils import comm
from AdaMatcher.src.utils.augment import build_augmentor
from AdaMatcher.src.utils.dataloader import get_local_split
from AdaMatcher.src.utils.misc import tqdm_joblib


class MultiSceneDataModule(pl.LightningDataModule):
    """For distributed training, each training process is assigned only a part
    of the training scenes to reduce memory overhead."""
    def __init__(self, args, config):
        super().__init__()

        self.dataset_cfg = config.DATASET

        # 1. data config
        # Train and Val should from the same data source
        self.train_data_source = config.DATASET.TRAIN_DATA_SOURCE
        self.val_data_source = config.DATASET.VAL_DATA_SOURCE
        self.test_data_source = config.DATASET.TEST_DATA_SOURCE
        # training and validating
        self.train_data_root = config.DATASET.TRAIN_DATA_ROOT
        self.train_walkdepth = config.DATASET.TRAIN_WALKDEPTH
        self.train_pose_root = config.DATASET.TRAIN_POSE_ROOT  # (optional)
        self.train_npz_root = config.DATASET.TRAIN_NPZ_ROOT
        self.train_list_path = config.DATASET.TRAIN_LIST_PATH
        self.train_intrinsic_path = config.DATASET.TRAIN_INTRINSIC_PATH
        self.val_data_root = config.DATASET.VAL_DATA_ROOT
        self.val_walkdepth = config.DATASET.VAL_WALKDEPTH
        self.val_pose_root = config.DATASET.VAL_POSE_ROOT  # (optional)
        self.val_npz_root = config.DATASET.VAL_NPZ_ROOT
        self.val_list_path = config.DATASET.VAL_LIST_PATH
        self.val_intrinsic_path = config.DATASET.VAL_INTRINSIC_PATH
        # testing
        self.test_data_root = config.DATASET.TEST_DATA_ROOT
        self.test_walkdepth = config.DATASET.TEST_WALKDEPTH
        self.test_pose_root = config.DATASET.TEST_POSE_ROOT  # (optional)
        self.test_npz_root = config.DATASET.TEST_NPZ_ROOT
        self.test_list_path = config.DATASET.TEST_LIST_PATH
        self.test_intrinsic_path = config.DATASET.TEST_INTRINSIC_PATH

        # 2. dataset config
        # general options
        self.min_overlap_score_test = (
            config.DATASET.MIN_OVERLAP_SCORE_TEST
        )  # 0.4, omit data with overlap_score < min_overlap_score
        self.min_overlap_score_train = config.DATASET.MIN_OVERLAP_SCORE_TRAIN
        self.augment_fn = build_augmentor(
            config.DATASET.AUGMENTATION_TYPE
        )  # None, options: [None, 'dark', 'mobile']

        # MegaDepth options
        self.mgdpt_img_resize = config.DATASET.MGDPT_IMG_RESIZE  # 840
        self.mgdpt_img_pad = config.DATASET.MGDPT_IMG_PAD  # True
        self.mgdpt_depth_pad = config.DATASET.MGDPT_DEPTH_PAD  # True
        self.mgdpt_df = config.DATASET.MGDPT_DF  # 8
        self.geometric_augmentation = config.DATASET.GEOMETRIC_AUGMENTATION 
        self.coarse_scale = (1 / config.ADAMATCHER.RESOLUTION[0]
                             )  # 0.125. for training adamatcher.

        # 3.loader parameters
        self.train_loader_params = {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True),
        }
        self.val_loader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True),
        }
        self.test_loader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': args.num_workers,
            'pin_memory': True,
        }

        # 4. sampler
        self.data_sampler = config.TRAINER.DATA_SAMPLER
        self.n_samples_per_subset = config.TRAINER.N_SAMPLES_PER_SUBSET
        self.subset_replacement = config.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT
        self.shuffle = config.TRAINER.SB_SUBSET_SHUFFLE
        self.repeat = config.TRAINER.SB_REPEAT

        # (optional) RandomSampler for debugging

        # misc configurations
        self.parallel_load_data = getattr(args, 'parallel_load_data', False)
        self.seed = config.TRAINER.SEED  # 66

    def setup(self, stage=None):
        """Setup train / val / test dataset.

        This method will be called by PL automatically.
        Args:
            stage (str): 'fit' in training phase, and 'test' in testing phase.
        """

        assert stage in ['fit', 'test'], 'stage must be either fit or test'

        # try:
        #     self.world_size = dist.get_world_size()
        #     self.rank = dist.get_rank()
        #     logger.info(f'[rank:{self.rank}] world_size: {self.world_size}')
        # except AssertionError as ae:
        self.world_size = 1
        self.rank = 0

        if stage == 'fit':
            self.train_dataset = self._setup_dataset(
                self.train_data_root,
                self.train_npz_root,
                self.train_list_path,
                self.train_intrinsic_path,
                mode='train',
                data_source=self.train_data_source,
                min_overlap_score=self.min_overlap_score_train,
                pose_dir=self.train_pose_root,
                walk_depth=self.train_walkdepth,
            )
            # setup multiple (optional) validation subsets
            if isinstance(self.val_list_path, (list, tuple)):
                self.val_dataset = []
                if not isinstance(self.val_npz_root, (list, tuple)):
                    self.val_npz_root = [
                        self.val_npz_root
                        for _ in range(len(self.val_list_path))
                    ]
                for npz_list, npz_root in zip(self.val_list_path,
                                              self.val_npz_root):
                    self.val_dataset.append(
                        self._setup_dataset(
                            self.val_data_root,
                            npz_root,
                            npz_list,
                            self.val_intrinsic_path,
                            mode='val',
                            data_source=self.val_data_source,
                            min_overlap_score=self.min_overlap_score_test,
                            pose_dir=self.val_pose_root,
                            walk_depth=self.val_walkdepth
                        ))
            else:
                self.val_dataset = self._setup_dataset(
                    self.val_data_root,
                    self.val_npz_root,
                    self.val_list_path,
                    self.val_intrinsic_path,
                    mode='val',
                    data_source=self.val_data_source,
                    min_overlap_score=self.min_overlap_score_test,
                    pose_dir=self.val_pose_root,
                    walk_depth=self.val_walkdepth,
                )
            logger.info(f'[rank:{self.rank}] Train & Val Dataset loaded!')
        else:  # stage == 'test
            self.test_dataset = self._setup_dataset(
                self.test_data_root,
                self.test_npz_root,
                self.test_list_path,
                self.test_intrinsic_path,
                mode='test',
                data_source=self.test_data_source,
                min_overlap_score=self.min_overlap_score_test,
                pose_dir=self.test_pose_root,
                walk_depth=self.test_walkdepth,
            )
            logger.info(f'[rank:{self.rank}]: Test Dataset loaded!')

    def _setup_dataset(
        self,
        data_root,
        split_npz_root,
        scene_list_path,
        intri_path,
        data_source,
        mode='train',
        min_overlap_score=0.0,
        pose_dir=None,
        walk_depth=False,
    ):
        """Setup train / val / test set."""
        with open(scene_list_path, 'r') as f:
            npz_names = [name.split()[0] for name in f.readlines()]

        if mode == 'train':
            local_npz_names = get_local_split(npz_names, self.world_size,
                                              self.rank, self.seed)
        else:
            local_npz_names = npz_names
        logger.info(
            f'[rank {self.rank}]: {len(local_npz_names)} scene(s) assigned.')

        dataset_builder = (self._build_concat_dataset_parallel
                           if self.parallel_load_data else
                           self._build_concat_dataset)

        return dataset_builder(
            data_root,
            local_npz_names,
            split_npz_root,
            intri_path,
            mode=mode,
            data_source=data_source,
            min_overlap_score=min_overlap_score,
            pose_dir=pose_dir,
            walk_depth=walk_depth,
        )

    def _build_concat_dataset(
        self,
        data_root,
        npz_names,
        npz_dir,
        intrinsic_path,
        mode,
        data_source,
        min_overlap_score=0.0,
        pose_dir=None,
        walk_depth=False,
    ):
        datasets = []
        augment_fn = self.augment_fn if mode == 'train' else None
        # data_source = (self.trainval_data_source
        #                if mode in ['train', 'val'] else self.test_data_source)
        if str(data_source).lower() == 'megadepth':
            npz_names = [f'{n}.npz' for n in npz_names]

        for npz_name in tqdm(
                npz_names,
                desc=f'[rank:{self.rank}] loading {mode} datasets',
                disable=int(self.rank) != 0,
        ):
            # `ScanNetDataset`/`MegaDepthDataset` load all data from npz_path when initialized, which might take time.
            npz_path = osp.join(npz_dir, npz_name)
            if data_source == 'ScanNet':
                datasets.append(
                    ScanNetDataset(
                        data_root,
                        npz_path,
                        intrinsic_path,
                        mode=mode,
                        min_overlap_score=min_overlap_score,
                        augment_fn=augment_fn,
                        pose_dir=pose_dir,
                    ))
            elif data_source == 'MegaDepth':
                if os.path.exists(npz_path) == False:
                    logger.warning(f'npz_path {npz_path} does not exist!')
                    continue
                datasets.append(
                    MegaDepthDataset(
                        data_root,
                        npz_path,
                        mode=mode,
                        min_overlap_score=min_overlap_score,
                        img_resize=self.mgdpt_img_resize,
                        df=self.mgdpt_df,
                        img_padding=self.mgdpt_img_pad,
                        depth_padding=self.mgdpt_depth_pad,
                        augment_fn=augment_fn,
                        coarse_scale=self.coarse_scale,
                        walk_depth=walk_depth,
                        geometric_augmentation=self.geometric_augmentation if mode == 'train' else False
                    ))
            else:
                raise NotImplementedError()
            
        return ConcatDataset([ds for ds in datasets if len(ds) > 0])

    def _build_concat_dataset_parallel(
        self,
        data_root,
        npz_names,
        npz_dir,
        intrinsic_path,
        mode,
        data_source,
        min_overlap_score=0.0,
        pose_dir=None,
        walk_depth=False,
    ):
        augment_fn = self.augment_fn if mode == 'train' else None
        # data_source = (self.trainval_data_source
        #                if mode in ['train', 'val'] else self.test_data_source)
        if str(data_source).lower() == 'megadepth':
            npz_names = [f'{n}.npz' for n in npz_names]
        with tqdm_joblib(
                tqdm(
                    desc=f'[rank:{self.rank}] loading {mode} datasets',
                    total=len(npz_names),
                    disable=int(self.rank) != 0,
                )):
            if data_source == 'ScanNet':
                datasets = Parallel(n_jobs=math.floor(
                    len(os.sched_getaffinity(0)) * 0.9 /
                    comm.get_local_size()))(delayed(lambda x: _build_dataset(
                        ScanNetDataset,
                        data_root,
                        osp.join(npz_dir, x),
                        intrinsic_path,
                        mode=mode,
                        min_overlap_score=min_overlap_score,
                        augment_fn=augment_fn,
                        pose_dir=pose_dir,
                    ))(name) for name in npz_names)
            elif data_source == 'MegaDepth':
                # TODO: _pickle.PicklingError: Could not pickle the task to send it to the workers.
                raise NotImplementedError()
                datasets = Parallel(n_jobs=math.floor(
                    len(os.sched_getaffinity(0)) * 0.9 /
                    comm.get_local_size()))(delayed(lambda x: _build_dataset(
                        MegaDepthDataset,
                        data_root,
                        osp.join(npz_dir, x),
                        mode=mode,
                        min_overlap_score=min_overlap_score,
                        img_resize=self.mgdpt_img_resize,
                        df=self.mgdpt_df,
                        img_padding=self.mgdpt_img_pad,
                        depth_padding=self.mgdpt_depth_pad,
                        augment_fn=augment_fn,
                        coarse_scale=self.coarse_scale,
                    ))(name) for name in npz_names)
            else:
                raise ValueError(f'Unknown dataset: {data_source}')
        return ConcatDataset([ds for ds in datasets if len(ds) > 0])

    def train_dataloader(self):
        """Build training dataloader for ScanNet / MegaDepth."""
        assert self.data_sampler in ['scene_balance']
        logger.info(
            f'[rank:{self.rank}/{self.world_size}]: Train Sampler and DataLoader re-init (should not re-init between epochs!).'
        )
        # if self.data_sampler == 'scene_balance':
        #     sampler = RandomConcatSampler(
        #         self.train_dataset,
        #         self.n_samples_per_subset,
        #         self.subset_replacement,
        #         self.shuffle,
        #         self.repeat,
        #         self.seed,
        #     )
        # else:
        #     sampler = None
        dataloader = DataLoader(self.train_dataset,
                                # sampler=sampler,
                                **self.train_loader_params)
        return dataloader

    def val_dataloader(self):
        """Build validation dataloader for ScanNet / MegaDepth."""
        logger.info(
            f'[rank:{self.rank}/{self.world_size}]: Val Sampler and DataLoader re-init.'
        )
        if not isinstance(self.val_dataset, abc.Sequence):
            # sampler = DistributedSampler(self.val_dataset, shuffle=False)
            return DataLoader(self.val_dataset,
                            #   sampler=sampler,
                              **self.val_loader_params)
        else:
            dataloaders = []
            for dataset in self.val_dataset:
                # sampler = DistributedSampler(dataset, shuffle=False)
                dataloaders.append(
                    DataLoader(dataset,
                            #    sampler=sampler,
                               **self.val_loader_params))
            return dataloaders

    def test_dataloader(self, *args, **kwargs):
        logger.info(
            f'[rank:{self.rank}/{self.world_size}]: Test Sampler and DataLoader re-init.'
        )
        # sampler = DistributedSampler(self.test_dataset, shuffle=False)
        # sampler = RandomConcatSampler(self.test_dataset,
        #                               self.n_samples_per_subset,
        #                               self.subset_replacement,
        #                               self.shuffle, self.repeat, self.seed)
        return DataLoader(self.test_dataset,
                        #   sampler=sampler,
                          **self.test_loader_params)


def _build_dataset(dataset: Dataset, *args, **kwargs):
    return dataset(*args, **kwargs)
