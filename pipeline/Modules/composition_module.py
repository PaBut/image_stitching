import glob
import os
from cv2 import Mat
import cv2
import numpy as np
from abc import ABC, abstractmethod
import torch
from pipeline.Modules.tools.UDIS2.Composition.network import Network, build_model

class CompositionModule(ABC):
    @abstractmethod
    def composite(self, src: Mat, src_mask: Mat, dst: Mat, dst_mask: Mat) -> tuple[Mat, Mat, Mat]:
        pass

def min(a, b):
    if a < b:
        return a
    return b

def max(a, b):
    if a > b:
        return a
    return b

class UdisCompositionModule(CompositionModule):
    MODEL_DIR = './pipeline/Modules/tools/UDIS2/Composition/model'
    MIN_DIM_SIZE=408
    def preprocessData(self, src: Mat, dst: Mat, src_mask: Mat, dst_mask: Mat):
        warp1 = src.astype(dtype=np.float32)
        warp1 = (warp1 / 127.5) - 1.0
        warp1 = np.transpose(warp1, [2, 0, 1])

        warp2 = dst.astype(dtype=np.float32)
        warp2 = (warp2 / 127.5) - 1.0
        warp2 = np.transpose(warp2, [2, 0, 1])

        mask1 = src_mask.astype(dtype=np.float32)
        mask1 = mask1 / 255
        mask1 = np.transpose(mask1, [2, 0, 1])

        mask2 = dst_mask.astype(dtype=np.float32)
        mask2 = mask2 / 255
        mask2 = np.transpose(mask2, [2, 0, 1])

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
        else:
            print('No checkpoint found!')
            return
        
        h, w, _ = src.shape
        src_copy = np.copy(src)
        src_mask_copy = np.copy(src_mask)
        dst_copy = np.copy(dst)
        dst_mask_copy = np.copy(dst_mask)

        resize_flag = min(h, w) < self.MIN_DIM_SIZE

        if resize_flag:
            print('resized')
            resize = max(self.MIN_DIM_SIZE / w, self.MIN_DIM_SIZE / h)

            new_h, new_w = int(h * resize), int(w * resize)

            src_copy = cv2.resize(src_copy, (new_w, new_h))
            src_mask_copy = cv2.resize(src_mask_copy, (new_w, new_h))
            dst_copy = cv2.resize(dst_copy, (new_w, new_h))
            dst_mask_copy = cv2.resize(dst_mask_copy, (new_w, new_h))
            print(src_copy.shape)

        src_tensor, dst_tensor, src_mask_tensor, dst_mask_tensor = self.preprocessData(src_copy, dst_copy, src_mask_copy, dst_mask_copy)
        
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

        stitched_image = ((stitched_image[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
        learned_mask1 = (learned_mask1[0]*255).cpu().detach().numpy().transpose(1,2,0)
        learned_mask2 = (learned_mask2[0]*255).cpu().detach().numpy().transpose(1,2,0)

        if resize_flag:
            stitched_image = cv2.resize(stitched_image, (w, h))
            learned_mask1 = cv2.resize(learned_mask1, (w, h))
            learned_mask2 = cv2.resize(learned_mask2, (w, h))

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
    
class WeightedAlphaCompositionModule(CompositionModule):
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
        





