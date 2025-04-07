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
    

class AachenDataModule(DataModule):
    def __init__(self, data_root: str, npz_path: str, min_overlap_score: float = 0.2):
        self.data_root = data_root
        self.scene_info = dict(np.load(npz_path, allow_pickle=True))

        self.files : list[tuple[str, str]] = []

        self.pair_infos = self.scene_info['pair_infos'].copy()

        self.pair_infos = [
            pair_info for pair_info in self.pair_infos
            if pair_info[1] > min_overlap_score
        ]

    def get_random_pair(self) -> tuple[Mat, Mat]:
        img_names = random.sample(self.pair_infos, 1)[0]

        (img1, img2), _, _ = img_names

        print(img1)
        print(img1)

        img1 = cv2.imread(os.path.join(self.data_root, img1))
        img2 = cv2.imread(os.path.join(self.data_root, img2))

        return img1, img2
    
    def get_n_random_pairs(self, n: int) -> list[tuple[Mat, Mat]]:
        n = min(len(self.files), n)
        pairs = random.sample(self.pair_infos, n)

        return [(cv2.imread(img_names[0][0]),
                  cv2.imread(img_names[0][1])) for img_names in pairs]


        
