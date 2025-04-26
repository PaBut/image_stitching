from abc import ABC, abstractmethod
from enum import Enum
import glob

from cv2 import Mat
import cv2
import numpy as np
import torch
import torchvision.transforms as T

resize_512 = T.Resize((512,512))

from enums import EnvironmentType
from metrics import MetricsCalculator
from warp_module import SingleWarpModule, WarpModule
from match_finders import AdaMatcherMatchFinder, FeatureDetector, FeatureDetectorMatchFinder, LoFTRMatchFinder, MatchFinder
from tools.UDIS2.Warp.network import Network, build_new_ft_model, get_stitched_result

class DetectorFreeModel(Enum):
    LoFTR = 0,
    AdaMatcher = 1

class Warper(ABC):
    matcher: MatchFinder
    warp_module: WarpModule
    def construct_warp(self, img0: Mat, img1: Mat, matching_metrics: dict | None = None) -> tuple[Mat, Mat, Mat, Mat] | None:
        keypoints1, keypoints2 = self.matcher.find_matches(img0, img1)
        if matching_metrics != None:
            metrics_calculator = MetricsCalculator()
            metrics = metrics_calculator.get_matching_metrics(keypoints1, keypoints2, [3, 5, 10])
            matching_metrics["auc"] = metrics["auc"]
        return self.warp_module.warp_images(img0, img1, keypoints1, keypoints2)

class UDIS2Warper(Warper):
    MODEL_DIR = './tools/UDIS2/Warp/model'
    def __init__(self):
        super().__init__()

    def __loadSingleData(self, img0: Mat, img1: Mat):
        input1 = img0.astype(dtype=np.float32)
        input1 = (input1 / 127.5) - 1.0
        input1 = np.transpose(input1, [2, 0, 1])
        input2 = img1.astype(dtype=np.float32)
        input2 = (input2 / 127.5) - 1.0
        input2 = np.transpose(input2, [2, 0, 1])
        # convert to tensor
        input1_tensor = torch.tensor(input1).unsqueeze(0)
        input2_tensor = torch.tensor(input2).unsqueeze(0)
        return (input1_tensor, input2_tensor)

    def construct_warp(self, img0, img1, matching_metrics: dict | None = None):
        net = Network()
        if torch.cuda.is_available():
            net = net.cuda()

        #load the existing models if it exists
        ckpt_list = glob.glob(self.MODEL_DIR + "/*.pth")
        ckpt_list.sort()
        if len(ckpt_list) != 0:
            model_path = ckpt_list[-1]
            checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint['model'])
            # print('load model from {}!'.format(model_path))
        else:
            print('No checkpoint found!')
            return

        print(img0.shape, img1.shape)
        input1_tensor, input2_tensor = self.__loadSingleData(img0, img1)
        if torch.cuda.is_available():
            input1_tensor = input1_tensor.cuda()
            input2_tensor = input2_tensor.cuda()
        
        input1_tensor_512 = resize_512(input1_tensor)
        input2_tensor_512 = resize_512(input2_tensor)

        batch_out = build_new_ft_model(net, input1_tensor_512, input2_tensor_512)
        warp_mesh = batch_out['warp_mesh']
        warp_mesh_mask = batch_out['warp_mesh_mask']
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
    def __init__(self, detector_type: FeatureDetector):
        self.matcher = FeatureDetectorMatchFinder(detector_type)
        self.warp_module = SingleWarpModule()
        super().__init__()
    

class DetectorFreeWarper(Warper):
    def __init__(self, model: DetectorFreeModel, model_type: EnvironmentType | None = None, model_path: str | None = None):
        if model == DetectorFreeModel.LoFTR:
            if model_type == None:
                raise Exception("Model type needs to be provided")
            self.matcher = LoFTRMatchFinder(model_type, model_path)
        if model == DetectorFreeModel.AdaMatcher:
            self.matcher = AdaMatcherMatchFinder(model_path)
        self.warp_module = SingleWarpModule()
        super().__init__()