import math
import os.path as osp
import pdb
import random
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import kornia.geometry.transform as KT
from loguru import logger
from torch.utils.data import Dataset
from functools import partial as partial

from src.utils.dataset import (read_megadepth_color, read_megadepth_depth,
                               read_megadepth_gray, read_scannet_color)


def get_rotation_matrix_2d(angle_degrees):
    theta = math.radians(angle_degrees)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    Rz = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta,  cos_theta, 0],
        [0,          0,         1]
    ], dtype=np.float32)
    
    return Rz

def apply_rotation_matrix_pose(pose, matrix):
    R = pose[:3, :3]
    t = pose[:3, 3]

    R_new = R @ matrix # Rotate camera
    pose_new = np.eye(4)
    pose_new[:3, :3] = R_new
    pose_new[:3, 3] = t
    return pose_new

def rotate_pose(pose, angle_deg):
    Rz = get_rotation_matrix_2d(angle_deg)
    
    return apply_rotation_matrix_pose(pose, Rz)

def apply_geometric_augmentation(image, mask, depth, K, T, scale_wh, scale, rotation: bool, hflip: bool, vflip: bool):
    if rotation:
        rotation = np.random.uniform(-30, 30)
        rotate = partial(KT.rotate, angle=torch.tensor([rotation], device=image.device),
                          center=torch.tensor([[scale_wh[0] / 2, scale_wh[1] / 2]],
                                               dtype=torch.float32, device=image.device))
        image = rotate(image.unsqueeze(0)).squeeze(0)
        depth = rotate(depth.unsqueeze(0)).squeeze(0)
        mask = rotate(mask.to(dtype=torch.float32).unsqueeze(0)).squeeze(0) > 0.5
        T = rotate_pose(T, rotation)

    if hflip:
        matrix = np.array([
            [-1,  0, 0],
            [ 0,  1, 0],
            [ 0,  0, 1]
        ], dtype=np.float32)

        T = apply_rotation_matrix_pose(T, matrix)

    if vflip:
        matrix = np.array([
            [1,  0, 0],
            [0, -1, 0],
            [0,  0, 1]
        ], dtype=np.float32)

        T = apply_rotation_matrix_pose(T, matrix)

    return image, mask, depth, K, T


class MegaDepthDataset(Dataset):
    def __init__(self,
                 root_dir,
                 npz_path,
                 mode='train',
                 min_overlap_score=0.4,
                 img_resize=None,
                 df=None,
                 img_padding=False,
                 depth_padding=False,
                 augment_fn=None,
                 **kwargs):
        """Manage one scene(npz_path) of MegaDepth dataset.

        Args:
            root_dir (str): megadepth root directory that has `phoenix`.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            mode (str): options are ['train', 'val', 'test']
            min_overlap_score (float): how much a pair should have in common. In range of [0, 1]. Set to 0 when testing.
            img_resize (int, optional): the longer edge of resized images. None for no resize. 640 is recommended.
                                        This is useful during training with batches and testing with memory intensive algorithms.
            df (int, optional): image size division factor. NOTE: this will change the final image size after img_resize.
            img_padding (bool): If set to 'True', zero-pad the image to squared size. This is useful during training.
            depth_padding (bool): If set to 'True', zero-pad depthmap to (2000, 2000). This is useful during training.
            augment_fn (callable, optional): augments images with pre-defined visual effects.
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.scene_id = npz_path.split('.')[0]

        # prepare scene_info and pair_info
        # pdb.set_trace()
        # if mode == 'test' and min_overlap_score != 0:
        #     logger.warning(
        #         'You are using `min_overlap_score`!=0 in test mode. Set to 0.')
        #     min_overlap_score = 0

        self.scene_info = dict(np.load(npz_path, allow_pickle=True))
        self.pair_infos = self.scene_info['pair_infos'].copy()
        # del self.scene_info['pair_infos']
        self.pair_infos = [
            pair_info for pair_info in self.pair_infos
            if pair_info[1] > min_overlap_score
        ]

        # parameters for image resizing, padding and depthmap padding
        if mode == 'train':
            assert img_resize is not None and img_padding and depth_padding
        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding
        self.depth_max_size = (
            2000 if depth_padding else None
        )  # the upperbound of depthmaps size in megadepth.

        # for training AdaMatcher
        self.augment_fn = augment_fn if mode == 'train' else None
        self.coarse_scale = kwargs[
            'coarse_scale']  # getattr(kwargs, 'coarse_scale', 0.125)
        self.is_walkdepth = kwargs.get('walk_depth', False)
        self.geometric_augmentation = kwargs.get(
            'geometric_augmentation', False)

        if self.is_walkdepth:
            self.root_dir = self.scene_id

    def __len__(self):
        return len(self.pair_infos)

    def __getitem__(self, idx):
        (idx0, idx1), overlap_score, central_matches = self.pair_infos[idx]
        # (idx1, idx0), overlap_score, central_matches = self.pair_infos[idx]

        # read grayscale image and mask. (1, h, w) and (h, w)
        img_name0 = osp.join(self.root_dir,
                             self.scene_info['image_paths'][idx0])
        img_name1 = osp.join(self.root_dir,
                             self.scene_info['image_paths'][idx1])
        
        if self.geometric_augmentation:
            hflip0=np.random.choice([True, False], p=[0.25, 0.75])
            hflip1=np.random.choice([True, False], p=[0.25, 0.75])

            vflip0=np.random.choice([True, False], p=[0.02, 0.98])
            vflip1=np.random.choice([True, False], p=[0.02, 0.98])

        else:
            hflip0=False
            hflip1=False

            vflip0=False
            vflip1=False

        # TODO: Support augmentation & handle seeds for each worker correctly.
        # if 'rots' in self.scene_info and 0:
        #     rot0, rot1 = self.scene_info['rots'][idx]
        # else:
        rot0, rot1 = 0, 0
        if 'yfcc' in img_name0:
            image0, mask0, scale0, scale_wh0 = read_scannet_color(
                img_name0, resize=(640, 480), augment_fn=None, rotation=rot0)
            image1, mask1, scale1, scale_wh1 = read_scannet_color(
                img_name1, resize=(640, 480), augment_fn=None, rotation=rot1)
        else:
            image0, mask0, scale0, scale_wh0 = read_megadepth_color(
                img_name0, self.img_resize, self.df, self.img_padding,
            np.random.choice([self.augment_fn, None], p=[0.6, 0.4]), hflip=hflip0, vflip=vflip0)
            image1, mask1, scale1, scale_wh1 = read_megadepth_color(
                img_name1, self.img_resize, self.df, self.img_padding, 
            np.random.choice([self.augment_fn, None], p=[0.6, 0.4]), hflip=hflip1, vflip=vflip1)
        # read depth. shape: (h, w)
        if self.mode in ['train', 'val']:
            depth0 = read_megadepth_depth(
                osp.join(self.root_dir, self.scene_info['depth_paths'][idx0]),
                pad_to=self.depth_max_size,
                hflip=hflip0, vflip = vflip0
            )
            depth1 = read_megadepth_depth(
                osp.join(self.root_dir, self.scene_info['depth_paths'][idx1]),
                pad_to=self.depth_max_size,
                hflip=hflip1, vflip = vflip1
            )
        else:
            depth0 = depth1 = torch.tensor([])


        # read intrinsics of original size
        K_0 = torch.tensor(self.scene_info['intrinsics'][idx0].copy(),
                           dtype=torch.float).reshape(3, 3)
        K_1 = torch.tensor(self.scene_info['intrinsics'][idx1].copy(),
                           dtype=torch.float).reshape(3, 3)

        # read and compute relative poses
        T0 = self.scene_info['poses'][idx0]
        T1 = self.scene_info['poses'][idx1]
                
        if self.geometric_augmentation:
            image0, mask0, depth0, K_0, T0 = apply_geometric_augmentation(
                image0, mask0, depth0, K_0, T0, scale_wh0, scale0,
                rotation=np.random.choice([True, False], p=[0.25, 0.75]), 
                hflip=hflip0,
                vflip=vflip0)
            
        if self.geometric_augmentation:
            image1, mask1, depth1, K_1, T1 = apply_geometric_augmentation(
                image1, mask1, depth1, K_1, T1, scale_wh1, scale1,
                rotation=np.random.choice([True, False], p=[0.25, 0.75]), 
                hflip=hflip1,
                vflip=vflip1)

        T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)),
                              dtype=torch.float)[:4, :4]  # (4, 4)
        T_1to0 = T_0to1.inverse()

        data = {
            'image0':
            image0,  # (3, h, w)
            'depth0':
            depth0,  # (h, w)
            'image1':
            image1,
            'depth1':
            depth1,
            'T_0to1':
            T_0to1,  # (4, 4)
            'T_1to0':
            T_1to0,
            'K0':
            K_0,  # (3, 3)
            'K1':
            K_1,
            'scale0':
            scale0,  # [scale_w, scale_h]
            'scale1':
            scale1,
            'scale_wh0':
            scale_wh0,
            'scale_wh1':
            scale_wh1,
            'dataset_name':
            'MegaDepth',
            'scene_id':
            self.scene_id,
            'pair_id':
            idx,
            'pair_names': (
                self.scene_info['image_paths'][idx0],
                self.scene_info['image_paths'][idx1],
            ),
        }

        if mask0 is not None:
            [mask0_d8, mask1_d8] = F.interpolate(
                torch.stack([mask0, mask1], dim=0)[None].float(),
                scale_factor=1 / 8,
                mode='nearest',
                recompute_scale_factor=False,
            )[0].bool()
            [mask0_l0, mask1_l0] = F.interpolate(
                torch.stack([mask0, mask1], dim=0)[None].float(),
                scale_factor=self.coarse_scale,  # 1 / 64,
                mode='nearest',
                recompute_scale_factor=False,
            )[0].bool()
            data.update({
                'mask0_l0': mask0_l0,
                'mask1_l0': mask1_l0,
                'mask0_d8': mask0_d8,
                'mask1_d8': mask1_d8,
            })
        return data
