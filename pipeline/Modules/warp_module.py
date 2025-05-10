from cv2 import Mat
import cv2
import numpy as np
from abc import ABC, abstractmethod


class WarpModule(ABC):
    @abstractmethod
    def warp_images(self, img1: Mat, img2: Mat, keypoints1: np.array, keypoints2: np.array) -> tuple[Mat, Mat, Mat, Mat] | None:
        pass

class SingleWarpModule(WarpModule):
    REQUIRED_KEYPOINTS_COUNT = 4
    def warp_images(self, img1, img2, keypoints1, keypoints2):
        if len(keypoints1) < self.REQUIRED_KEYPOINTS_COUNT:
            return None
        
        H, _ = cv2.findHomography(np.array(keypoints2), np.array(keypoints1), cv2.RANSAC, 5.0)

        if H is None:
            return None

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

        warped_corners = cv2.perspectiveTransform(corners2, H)

        all_corners = np.concatenate(([[[0, 0]], [[0, h1]], [[w1, h1]], [[w1, 0]]], warped_corners), axis=0)

        # Find the bounding box of the final stitched image
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        # Translate the homography to adjust for any negative coordinates
        translation_dist = [-x_min, -y_min]

        H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

        dst = cv2.warpPerspective(img2, H_translation @ H, (x_max - x_min, y_max - y_min))
        
        canvas = np.zeros(dst.shape, np.uint8)
        mask2 = np.copy(canvas)
        cv2.fillPoly(mask2, [(warped_corners + translation_dist).astype(np.int32)], (255, 255, 255))
        cv2.polylines(mask2, [(warped_corners + translation_dist).astype(np.int32)], isClosed=True, color=(0, 0, 0), thickness=1)

        src = np.zeros(dst.shape, dst.dtype)
        src[int(translation_dist[1]):int(translation_dist[1] + h1), int(translation_dist[0]):int(translation_dist[0] + w1)] = img1

        mask1 = np.copy(canvas)
        mask1[int(translation_dist[1]):int(translation_dist[1] + h1), int(translation_dist[0]):int(translation_dist[0] + w1)] = 255

        return src, dst, mask1, mask2
