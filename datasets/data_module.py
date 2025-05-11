# @Author: Pavlo Butenko

import os
from abc import ABC, abstractmethod
import random
import re
import cv2
from cv2 import Mat
import numpy as np

class DataModule(ABC):
    """
    Abstract class for data modules.
    """
    @abstractmethod
    def get_random_pair(self) -> dict:
        """
        Returns a random pair of images with their ground-truth info.

        Returns:
            A dictionary containing:
                - img1: The first image.
                - img2: The second image.
                - depth_map1: The depth map of the first image.
                - depth_map2: The depth map of the second image.
                - K0: Intrinsic matrix of the first camera.
                - K1: Intrinsic matrix of the second camera.
                - T_0to1: Transformation matrix from camera 0 to camera 1.
                - T_1to0: Transformation matrix from camera 1 to camera 0.
        """
        pass

    @abstractmethod
    def get_n_random_pairs(self, n: int) -> list[dict]:
        """
        Returns n random pairs of images with their ground-truth info.

        Returns:
            A list of dictionaries, each containing:
                - img1: The first image.
                - img2: The second image.
                - depth_map1: The depth map of the first image.
                - depth_map2: The depth map of the second image.
        """
        pass
    
class WalkDepthDataLoader(DataModule):
    """
        Data module for the WalkDepth dataset made for visualization purposes.
    """
    def __init__(self, data_root: str, scene_list, min_overlap_score: float = 0.15, max_overlap_score: float = 1.0):
        """
        WalkDepthDataLoader constructor.

        Arguments:
            data_root (str): Path to the root directory of the dataset.
            scene_list (str): Path to the file containing the list of scenes.
            min_overlap_score (float): Minimal overlap score for image pairs.
            max_overlap_score (float): Maximal overlap score for image pairs.
        """
        with open(scene_list, 'r') as f:
            self.scene_list = f.readlines()
        self.scene_list = [line.strip() for line in self.scene_list]

        self.modules = []

        for scene in self.scene_list:
            scene_path = os.path.join(data_root, scene)
            npz_path = scene_path + ".npz"

            if not os.path.exists(npz_path):
                print(f"File not found: {npz_path}")
                continue

            module = WalkDepthDataModule(scene_path, npz_path, min_overlap_score, max_overlap_score)
            if len(module.pair_infos) > 0:
                self.modules.append(module)

    def get_random_pair(self) -> dict:
        module = random.sample(self.modules, 1)[0]

        return module.get_random_pair()
    
    def length(self) -> int:
        return sum(len(module.pair_infos) for module in self.modules)
    
    def get_n_random_pairs(self, n: int) -> list[dict]:
        modules = random.sample(self.modules, n)

        return  [module.get_random_pair() for module in modules]

def read_depthmap(path: str) -> np.ndarray:
    """
        Reads depth map binary file.
    """
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()
        
class WalkDepthDataModule(DataModule):
    """
    Data module for a single scene from the WalkDepth dataset made for visualization purposes.
    """
    def __init__(self, data_root: str, npz_path: str, min_overlap_score: float = 0.15, max_overlap_score: float = 1.0):
        """
        WalkDepthDataModule constructor.

        Arguments:
            data_root (str): Path to the root directory of the dataset.
            npz_path (str): Path to the .npz file containing scene information.
            min_overlap_score (float): Minimal overlap score for image pairs.
            max_overlap_score (float): Maximal overlap score for image pairs.
        """
        self.data_root = data_root
        self.scene_info = dict(np.load(npz_path, allow_pickle=True))

        self.pair_infos = self.scene_info['pair_infos'].copy()
        self.poses = self.scene_info['poses'].copy()
        self.intrinsics = self.scene_info['intrinsics'].copy()
        
        self.depth_maps = self.scene_info['depth_paths'].copy()
        self.files = self.scene_info['image_paths'].copy()

        self.pair_infos = [
            pair_info for pair_info in self.pair_infos
            if pair_info[1] > min_overlap_score and pair_info[1] < max_overlap_score
        ]

    def get_random_pair(self) -> dict:
        img_names = random.sample(self.pair_infos, 1)[0]

        (idx1, idx2), _, _ = img_names

        img1 = cv2.imread(os.path.join(self.data_root, self.files[idx1]))
        img2 = cv2.imread(os.path.join(self.data_root, self.files[idx2]))

        T_0to1 = np.matmul(self.poses[idx2], np.linalg.inv(self.poses[idx2]))[:4, :4]

        return {
            'img1': img1,
            'img2': img2,
            'depth_map1': read_depthmap(os.path.join(self.data_root, self.depth_maps[idx1])),
            'depth_map2': read_depthmap(os.path.join(self.data_root, self.depth_maps[idx2])),
            'K0' : self.intrinsics[idx1],
            'K1' : self.intrinsics[idx2],
            'T_0to1': T_0to1,
            'T_1to0': np.linalg.inv(T_0to1)
        }
    
    def get_n_random_pairs(self, n: int) -> list[dict]:
        n = min(len(self.files), n)
        pairs = random.sample(self.pair_infos, n)

        return [{
            "img1": cv2.imread(os.path.join(self.data_root, self.files[img_names[0][0]])),
            "img2": cv2.imread(os.path.join(self.data_root, self.files[img_names[0][1]])),
            "depth_map1": read_depthmap(os.path.join(self.data_root, self.depth_maps[img_names[0][0]])),
            "depth_map2": read_depthmap(os.path.join(self.data_root, self.depth_maps[img_names[0][1]]))
                  } for img_names in pairs]