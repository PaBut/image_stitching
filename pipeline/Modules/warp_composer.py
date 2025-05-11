# @Author: Pavlo Butenko

from abc import ABC, abstractmethod
from enum import Enum
import glob
import os

from cv2 import Mat
import numpy as np
import torch
import torchvision.transforms as T

resize_512 = T.Resize((512,512))

from pipeline.enums import EnvironmentType
from pipeline.Modules.warp_module import SingleWarpModule, WarpModule
from pipeline.Modules.match_finders import AdaMatcherMatchFinder, FeatureDetector, FeatureDetectorMatchFinder, LoFTRMatchFinder, MatchFinder
from models.UDIS2.Warp.network import Network, build_new_ft_model, get_stitched_result

class DetectorFreeModel(Enum):
    """
    Enum for deep learning models for feature matching
    """
    LoFTR = 0,
    AdaMatcher = 1

class Warper(ABC):
    """
    Abstract class for warping images.
    """
    matcher: MatchFinder
    warp_module: WarpModule
    def construct_warp(self, img0: Mat, img1: Mat) -> tuple[Mat, Mat, Mat, Mat] | None:
        """
            Warps two images into a shared perspective.

            Arguments:
                img0: The first image to warp.
                img1: The second image to warp.

            Returns:
                A tuple containing:
                    src: Canvas with img0 placed at its correct position.
                    dst: Canvas with img1 warped into the same space.
                    mask1: Binary mask indicating the region occupied by img0 on the canvas.
                    mask2: Binary mask indicating the region occupied by warped img1 on the canvas.

                None if the process fails.
        """
        keypoints1, keypoints2 = self.matcher.find_matches(img0, img1)

        return self.warp_module.warp_images(img0, img1, keypoints1, keypoints2)

class UDIS2Warper(Warper):
    """
    Class for warping images into shared perspective using the UDIS++ model.
    """
    MODEL_PATH = './models/UDIS2/Warp/model/epoch100_model.pth'
    def __init__(self):
        super().__init__()

    def __loadSingleData(self, img0: Mat, img1: Mat):
        """
        Preprocesses the input images for the model.
        """
        input1 = img0.astype(dtype=np.float32)
        input1 = (input1 / 127.5) - 1.0
        input1 = np.transpose(input1, [2, 0, 1])
        input2 = img1.astype(dtype=np.float32)
        input2 = (input2 / 127.5) - 1.0
        input2 = np.transpose(input2, [2, 0, 1])

        input1_tensor = torch.tensor(input1).unsqueeze(0)
        input2_tensor = torch.tensor(input2).unsqueeze(0)
        return input1_tensor, input2_tensor

    def construct_warp(self, img0, img1):
        net = Network()

        if os.path.exists(self.MODEL_PATH):
            checkpoint = torch.load(self.MODEL_PATH)
            net.load_state_dict(checkpoint['model'])
        else:
            print('No weights found for UDIS++ warping module!')
            return None

        if torch.cuda.is_available():
            net = net.cuda()

        input1_tensor, input2_tensor = self.__loadSingleData(img0, img1)
        if torch.cuda.is_available():
            input1_tensor = input1_tensor.cuda()
            input2_tensor = input2_tensor.cuda()
        
        input1_tensor_512 = resize_512(input1_tensor)
        input2_tensor_512 = resize_512(input2_tensor)

        batch_out = build_new_ft_model(net, input1_tensor_512, input2_tensor_512)
        rigid_mesh = batch_out['rigid_mesh']
        mesh = batch_out['mesh']
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
        with torch.no_grad():
            output = get_stitched_result(input1_tensor, input2_tensor, rigid_mesh, mesh)

        return (output['warp1'][0].cpu().detach().numpy().transpose(1,2,0),
                 output['warp2'][0].cpu().detach().numpy().transpose(1,2,0), 
                 output['mask1'][0].cpu().detach().numpy().transpose(1,2,0), 
                 output['mask2'][0].cpu().detach().numpy().transpose(1,2,0))

class FeatureDetectorWarper(Warper):
    """
    Class for warping images into shared perspective 
    using feature matcher based on handcrafted feature detectors.
    """
    def __init__(self, detector_type: FeatureDetector):
        """
        FeatureDetectorWarper constructor.

        Arguments:
            detector_type: The type of handcrafted feature detector to use.
        """
        self.matcher = FeatureDetectorMatchFinder(detector_type)
        self.warp_module = SingleWarpModule()
        super().__init__()


class DetectorFreeWarper(Warper):
    """
    Class for warping images into shared perspective based on deep learning feature matchers.
    """
    def __init__(self, model: DetectorFreeModel, model_type: EnvironmentType | None = None, model_path: str | None = None):
        """
        DetectorFreeWarper constructor.
        
        Arguments:
            model: The type of deep learning model to use for matching.
            model_type: The environment type for the LoFTR model (if applicable).
            model_path: The path to the model weights, if not supplied default path is used.
        """
        if model == DetectorFreeModel.LoFTR:
            if model_type == None:
                raise Exception("Model type needs to be provided")
            self.matcher = LoFTRMatchFinder(model_type, model_path)
        if model == DetectorFreeModel.AdaMatcher:
            self.matcher = AdaMatcherMatchFinder(model_path)
        self.warp_module = SingleWarpModule()
        super().__init__()