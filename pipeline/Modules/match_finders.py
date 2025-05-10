from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from cv2 import Mat
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pipeline.common import prepare_image
from pipeline.enums import EnvironmentType
from AdaMatcher.src.adamatcher.utils.cvpr_ds_config import lower_config
from AdaMatcher.src.adamatcher.adamatcher import AdaMatcher
from AdaMatcher.src.config.default import get_cfg_defaults
from models.LoFTR import LoFTR, default_cfg

import cv2
import numpy as np

class FeatureDetector(Enum):
    ORB = 0
    SIFT = 1
    BRISK = 2
    AKAZE = 3

class MatchFinder(ABC):
    @abstractmethod
    def find_matches(self, img1: Mat, img2: Mat) -> tuple[list[int], list[int]]:
        pass

class FeatureDetectorMatchFinder(MatchFinder):    
    def __init__(self, detector_type: FeatureDetector):
        if detector_type == FeatureDetector.SIFT:
            self.detector = cv2.SIFT.create()
        elif detector_type == FeatureDetector.AKAZE:
            self.detector = cv2.AKAZE.create()
        elif detector_type == FeatureDetector.BRISK:
            self.detector = cv2.BRISK.create()
        elif detector_type == FeatureDetector.ORB:
            self.detector = cv2.ORB.create()
        else:
            raise Exception('Not supported handcrafted method')

        self.matcher = cv2.BFMatcher()

    def find_matches(self, img1, img2):
        img1 = img1.astype(np.uint8)
        img2 = img2.astype(np.uint8)
        keypoints1, descriptors1 = self.detector.detectAndCompute(img1, None)
        keypoints2, descriptors2 = self.detector.detectAndCompute(img2, None)

        matches = self.matcher.knnMatch(descriptors2, descriptors1, k = 2)

        good_matches = []

        for m,n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        src_pts = np.float32([keypoints1[m.trainIdx].pt for m in good_matches])
        dst_pts = np.float32([keypoints2[m.queryIdx].pt for m in good_matches])

        if len(src_pts) == 0 or len(dst_pts) == 0:
            return np.empty((1, 2), dtype=float), np.empty((1, 2), dtype=float)

        return src_pts, dst_pts
        

class LoFTRMatchFinder(MatchFinder):
    INDOOR_WEIGHTS_PATH = './models/LoFTR/weights/indoor_ds_new.ckpt'
    OUTDOOR_WEIGHTS_PATH = './models/LoFTR/weights/outdoor_ds.ckpt'
    FIXED_WIDTH = 640
    def __init__(self, loftr_type: EnvironmentType, pretrained_ckpt=None):
        _default_cfg = deepcopy(default_cfg)
        self.DF = _default_cfg['resolution'][0]
        
        if loftr_type == EnvironmentType.Indoor:
            _default_cfg['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt
            _default_cfg['match_coarse']['match_type'] = 'dual_softmax'
            weights_path = self.INDOOR_WEIGHTS_PATH
        elif loftr_type == EnvironmentType.Outdoor:
            weights_path = self.OUTDOOR_WEIGHTS_PATH
            # _default_cfg['match_coarse']['match_type'] = 'sinkhorn'
            # _default_cfg['match_coarse']['sparse_spvs'] = False
        
        if pretrained_ckpt is not None:
            weights_path = pretrained_ckpt
        
        matcher = LoFTR(config=_default_cfg)
        matcher.load_state_dict(torch.load(weights_path)['state_dict'])
        self.matcher = matcher.eval()

        if torch.cuda.is_available():
            self.matcher = self.matcher.cuda()

    def find_matches(self, img0, img1):
        input0, resize0 = prepare_image(img0, self.DF)
        input1, resize1 = prepare_image(img1, self.DF)

        img1_torch = torch.from_numpy(cv2.cvtColor(input0, cv2.COLOR_RGB2GRAY))[None][None] / 255.
        img2_torch = torch.from_numpy(cv2.cvtColor(input1, cv2.COLOR_RGB2GRAY))[None][None] / 255.

        if torch.cuda.is_available():
            img1_torch = img1_torch.cuda()
            img2_torch = img2_torch.cuda()

        batch = {'image0': img1_torch, 'image1': img2_torch}
        with torch.no_grad():
            self.matcher(batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()

            return mkpts0 * resize0, mkpts1 * resize1
        

class AdaMatcherMatchFinder(MatchFinder):
    FIXED_WIDTH = 640
    FIXED_DIVISION = 32
    WEIGHTS_PATH = r'./AdaMatcher/weights/adamatcher.ckpt'
    def __init__(self, pretrained_ckpt=None):
        config = get_cfg_defaults()
        self.DF = config.DATASET.MGDPT_DF
        _config = lower_config(config)
        
        self.model = AdaMatcher(
            config = _config["adamatcher"],
            training=False
        )  
        weights_path = self.WEIGHTS_PATH
        if pretrained_ckpt is not None:
            weights_path = pretrained_ckpt

        torch.serialization.add_safe_globals([ModelCheckpoint])

        state_dict = torch.load(weights_path, weights_only=True)["state_dict"]
        
        new_state_dict = {}
        prefix = 'matcher.'
        for key, value in state_dict.items():
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        self.model.load_state_dict(new_state_dict)
        self.model = self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def find_matches(self, img1, img2):
        input1, resize1 = prepare_image(img1, self.DF)
        input2, resize2 = prepare_image(img2, self.DF)

        img1_torch = torch.from_numpy(cv2.cvtColor(input1, cv2.COLOR_BGR2RGB)
                                      .transpose(2, 0, 1))[None].float() / 255.
        img2_torch = torch.from_numpy(cv2.cvtColor(input2, cv2.COLOR_BGR2RGB)
                                      .transpose(2, 0, 1))[None].float() / 255.
        if torch.cuda.is_available():
            img1_torch = img1_torch.cuda()
            img2_torch = img2_torch.cuda()

        batch = {'image0': img1_torch, 'image1': img2_torch}
        with torch.no_grad():
            self.model(batch)

            pts0 = batch["mkpts0_f"].cpu().numpy()
            pts1 = batch["mkpts1_f"].cpu().numpy()

            return pts0 * resize1, pts1 * resize2