import glob
import os
from cv2 import Mat
import cv2
import numpy as np
from abc import ABC, abstractmethod
import torch
from tools.UDIS2.Composition.network import Network, build_model

class CompositionModule(ABC):
    @abstractmethod
    def composite(self, src: Mat, src_mask: Mat, dst: Mat, dst_mask: Mat) -> tuple[Mat, Mat, Mat]:
        pass

class UdisCompositionModule(CompositionModule):
    MODEL_DIR = './tools/UDIS2/Composition/model'
    def preprocessData(self, src: Mat, dst: Mat, src_mask: Mat, dst_mask: Mat):
        # load image1
        warp1 = src.astype(dtype=np.float32)
        warp1 = (warp1 / 127.5) - 1.0
        warp1 = np.transpose(warp1, [2, 0, 1])

        # load image2
        warp2 = dst.astype(dtype=np.float32)
        warp2 = (warp2 / 127.5) - 1.0
        warp2 = np.transpose(warp2, [2, 0, 1])

        # load mask1
        mask1 = src_mask.astype(dtype=np.float32)
        mask1 = mask1 / 255
        mask1 = np.transpose(mask1, [2, 0, 1])

        # load mask2
        mask2 = dst_mask.astype(dtype=np.float32)
        mask2 = mask2 / 255
        mask2 = np.transpose(mask2, [2, 0, 1])

        # convert to tensor
        warp1_tensor = torch.tensor(warp1).unsqueeze(0)
        warp2_tensor = torch.tensor(warp2).unsqueeze(0)
        mask1_tensor = torch.tensor(mask1).unsqueeze(0)
        mask2_tensor = torch.tensor(mask2).unsqueeze(0)

        return warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor

    def composite(self, src, src_mask, dst, dst_mask):
        os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        # define the network
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

        src_tensor, dst_tensor, src_mask_tensor, dst_mask_tensor = self.preprocessData(src, dst, src_mask, dst_mask)
        
        if torch.cuda.is_available():
            src_tensor = src_tensor.cuda()
            dst_tensor = dst_tensor.cuda()
            src_mask_tensor = src_mask_tensor.cuda()
            dst_mask_tensor = dst_mask_tensor.cuda()

        net.eval()
        with torch.no_grad():
            batch_out = build_model(net, src_tensor, dst_tensor, src_mask_tensor, dst_mask_tensor)
        stitched_image = batch_out['stitched_image']
        learned_mask1 = batch_out['learned_mask1']
        learned_mask2 = batch_out['learned_mask2']

        # (optional) draw composition images with different colors like our paper
        s1 = ((src_tensor[0]+1)*127.5 * learned_mask1[0]).cpu().detach().numpy().transpose(1,2,0)
        s2 = ((dst_tensor[0]+1)*127.5 * learned_mask2[0]).cpu().detach().numpy().transpose(1,2,0)
        fusion = np.zeros((src_tensor.shape[2],src_tensor.shape[3],3), np.uint8)
        fusion[...,0] = s2[...,0]
        fusion[...,1] = s1[...,1]*0.5 +  s2[...,1]*0.5
        fusion[...,2] = s1[...,2]

        # save learned masks and final composition
        stitched_image = ((stitched_image[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
        learned_mask1 = (learned_mask1[0]*255).cpu().detach().numpy().transpose(1,2,0)
        learned_mask2 = (learned_mask2[0]*255).cpu().detach().numpy().transpose(1,2,0)

        return stitched_image, learned_mask1, learned_mask2
    

class AlphaCompositionModule(CompositionModule):
    def composite(self, src, src_mask, dst, dst_mask):
        common_area = cv2.bitwise_and(src_mask, dst_mask) / 2

        mask1 = src_mask - common_area
        mask2 = dst_mask - common_area

        alpha1 = mask1 / 255
        alpha2 = mask2 / 255

        blended = src * alpha1 + dst * alpha2
        blended.astype(np.uint8)

        return blended, mask1, mask2
    
class ComplexAlphaCompositionModule(CompositionModule):
    def composite(self, src, src_mask, dst, dst_mask):
        common_area = cv2.bitwise_and(src_mask, dst_mask)

        contours, _ = cv2.findContours(cv2.cvtColor(src_mask.astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, None)
        x, y, w, h = cv2.boundingRect(contours[0])

        center = (x + w//2, y + h//2)

        x, y = np.meshgrid(np.arange(src.shape[1]), np.arange(src.shape[0]))
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        max_dist = np.sqrt((w/2)**2 + (h/2)**2)  # Maximum distance from center
        alpha_mask = np.expand_dims((1 - dist / max_dist), axis=-1)

        mask1 = src_mask - common_area + common_area * alpha_mask
        mask2 = dst_mask - common_area + common_area * (1 - alpha_mask)

        alpha1 = mask1 / 255
        alpha2 = mask2 / 255

        blended = src * alpha1 + dst * alpha2

        return blended, mask1, mask2

class SimpleCompositionModule(CompositionModule):
    def composite(self, src, src_mask, dst, dst_mask):
        common_area = cv2.bitwise_and(src_mask, dst_mask)

        mask2 = dst_mask - common_area

        return src * (src_mask / 255) + dst * (mask2 / 255), src_mask, mask2
        





