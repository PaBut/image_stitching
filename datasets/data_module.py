import os
from abc import ABC, abstractmethod
import random
import re
import cv2
from cv2 import Mat
import numpy as np


def get_files_from_directory(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def extract_number(file_name):
    match = re.search(r'\d+', file_name)
    return int(match.group()) if match else 0


class DataModule(ABC):
    @abstractmethod
    def get_random_pair(self) -> tuple[Mat, Mat]:
        pass

    @abstractmethod
    def get_n_random_pairs(self, n: int) -> list[tuple[Mat, Mat]]:
        pass


class UDISDataModule(DataModule):
    def __init__(self, path: str):
        self.dataset_path = path
        self.files = get_files_from_directory(os.path.join(path, 'input1'))

    def get_random_pair(self) -> tuple[Mat, Mat]:
        img_name = random.sample(self.files, 1)[0]

        print(img_name)

        img1 = cv2.imread(os.path.join(self.dataset_path, 'input1', img_name))
        img2 = cv2.imread(os.path.join(self.dataset_path, 'input2', img_name))

        return img1, img2
    
    def get_n_random_pairs(self, n: int) -> list[tuple[Mat, Mat]]:
        n = min(len(self.files), n)

        img_names = random.sample(self.files, n)

        return [(cv2.imread(os.path.join(self.dataset_path, 'input1', img_name)),
                  cv2.imread(os.path.join(self.dataset_path, 'input2', img_name))) for img_name in img_names]
    

class ISIQADataModule(DataModule):
    def __init__(self, path: str):
        self.dataset_path = path

        self.files : list[tuple[str, str]] = []

        scenes = os.listdir(path)

        for scene in scenes:
            dir_path = os.path.join(self.dataset_path, scene)

            files = get_files_from_directory(dir_path)

            files = sorted(files, key=extract_number)

            for i in range(len(files) - 1):
                self.files.append((os.path.join(dir_path, files[i]), os.path.join(dir_path, files[i + 1])))



    def get_random_pair(self) -> tuple[Mat, Mat]:
        img_names = random.sample(self.files, 1)[0]

        print(img_names[0])
        print(img_names[1])

        img1 = cv2.imread(img_names[0])
        img2 = cv2.imread(img_names[1])

        return img1, img2
    
    def get_n_random_pairs(self, n: int) -> list[tuple[Mat, Mat]]:
        n = min(len(self.files), n)
        img_names = random.sample(self.files, n)

        return [(cv2.imread(img_name[0]),
                  cv2.imread(img_name[1])) for img_name in img_names]
    
class WalkDepthDataLoader(DataModule):
    def __init__(self, data_root: str, scene_list, min_overlap_score: float = 0.2, max_overlap_score: float = 0.8):
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

    def get_random_pair(self) -> tuple[Mat, Mat]:
        module = random.sample(self.modules, 1)[0]

        return module.get_random_pair()
    
    def length(self) -> int:
        return sum(len(module.pair_infos) for module in self.modules)
    
    def get_n_random_pairs(self, n: int) -> list[tuple[Mat, Mat]]:
        modules = random.sample(self.modules, n)

        return  [module.get_random_pair() for module in modules]

def read_array(path):
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
    def __init__(self, data_root: str, npz_path: str, min_overlap_score: float = 0.2, max_overlap_score: float = 0.8):
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

    def get_random_pair(self) -> tuple[Mat, Mat]:
        img_names = random.sample(self.pair_infos, 1)[0]

        (idx1, idx2), _, _ = img_names

        img1 = cv2.imread(os.path.join(self.data_root, self.files[idx1]))
        img2 = cv2.imread(os.path.join(self.data_root, self.files[idx2]))

        print(os.path.join(self.data_root, self.depth_maps[idx1]))
        print(os.path.join(self.data_root, self.depth_maps[idx2]))
        T_0to1 = np.matmul(self.poses[idx2], np.linalg.inv(self.poses[idx2]))[:4, :4]

        return {
            'img1': img1,
            'img2': img2,
            'depth_map1': read_array(os.path.join(self.data_root, self.depth_maps[idx1])),
            'depth_map2': read_array(os.path.join(self.data_root, self.depth_maps[idx2])),
            'K0' : self.intrinsics[idx1],
            'K1' : self.intrinsics[idx2],
            'T_0to1': T_0to1,
            'T_1to0': np.linalg.inv(T_0to1)
        }
    
    def get_n_random_pairs(self, n: int) -> list[tuple[Mat, Mat]]:
        n = min(len(self.files), n)
        pairs = random.sample(self.pair_infos, n)

        return [{
            "img1": cv2.imread(os.path.join(self.data_root, self.files[img_names[0][0]])),
            "img2": cv2.imread(os.path.join(self.data_root, self.files[img_names[0][1]])),
            "depth_map1": read_array(os.path.join(self.data_root, self.depth_maps[img_names[0][0]])),
            "depth_map2": read_array(os.path.join(self.data_root, self.depth_maps[img_names[0][1]]))
                  } for img_names in pairs]