from cv2 import Mat
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim   
from sklearn import metrics
import pandas as pd

class Metrics:
    __is_initialized: bool = False

    composition_metrics: dict[str, float]
    warp_metrics: dict[str, float]
    matching_metrics: dict[str, float]
    time_taken: float

    PANDAS_COLUMNS = ["model", "metrics_type", "metrics_name", "value"]

    def metrics_entry_to_dict(self, model_name: str, metrics_type: str, metrics_name: str, value: float) -> dict[str, float]:
        return {self.PANDAS_COLUMNS[0]: model_name, self.PANDAS_COLUMNS[1]: metrics_type, 
                self.PANDAS_COLUMNS[2]: metrics_name, self.PANDAS_COLUMNS[3]: value}

    def to_pandas_df(self, model_name: str) -> pd.DataFrame: 
        dicts: list[dict[str, float]] = []

        for key, value in self.composition_metrics.items():
            dicts.append(self.metrics_entry_to_dict(model_name, "composition", key, value))

        for key, value in self.matching_metrics.items():
            dicts.append(self.metrics_entry_to_dict(model_name, "matching", key, value))

        for key, value in self.warp_metrics.items():
            dicts.append(self.metrics_entry_to_dict(model_name, "warp", key, value))

        dicts.append(self.metrics_entry_to_dict(model_name, "time", "time_taken", self.time_taken))

        return pd.DataFrame(dicts, columns=self.PANDAS_COLUMNS)
        

    def calculate_average_dict(self, src: dict[str, float], adding: dict[str, float]) -> dict[str, float]:
        result: dict[str, float] = {}
        for key in src.keys():
            result[key] = (src[key] + adding[key]) / 2

        return result

    def combine_metrics(self, metrics):
        if not self.__is_initialized:
            self.composition_metrics = metrics.composition_metrics
            self.warp_metrics = metrics.warp_metrics
            self.matching_metrics = metrics.matching_metrics
            self.time_taken = metrics.time_taken

            self.__is_initialized = True
        else: 
            self.composition_metrics = self.calculate_average_dict(self.composition_metrics, metrics.composition_metrics)
            self.warp_metrics = self.calculate_average_dict(self.warp_metrics, metrics.warp_metrics)
            self.matching_metrics = self.calculate_average_dict(self.matching_metrics, metrics.matching_metrics)
            self.time_taken = (self.time_taken + metrics.time_taken) / 2

class MetricsCalculator:
    def __calculate_ssim(self, img1: Mat, img2: Mat) -> float:
        return ssim(img1, img2, channel_axis=-1)

    def __calculate_mse(self, img1: Mat, img2: Mat) -> float:
        return np.mean((img1 - img2) ** 2)
    
    def __calculate_psnr(self, img1: Mat, img2: Mat) -> float:
        return cv2.PSNR(img1, img2)
    
    def __calculate_seam_score(self, overlapped_result: Mat) -> float:
        edges = cv2.Canny(overlapped_result, 100, 200)
        seam_score = np.sum(edges) / np.prod(overlapped_result.shape)
        return seam_score
    
    def get_composition_metrics(self, src_img: Mat, dst_img: Mat, composition_img: Mat, src_mask: Mat, dst_mask: Mat) -> dict[str, float]:
        src_img_cp = src_img.astype(np.uint8)
        dst_img_cp = dst_img.astype(np.uint8)
        composition_img_cp = composition_img.astype(np.uint8)

        overlap_mask = cv2.bitwise_and(src_mask, dst_mask).astype(np.uint8)

        contours, _ = cv2.findContours(cv2.cvtColor(overlap_mask, cv2.COLOR_RGB2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return {"ssim": 0, "mse" : 0, "psnr": 0}

        mainContour = contours[0]
        o_x, o_y, o_w, o_h = cv2.boundingRect(mainContour)

        if o_w < 7 or o_h < 7:
            return {"ssim": 0, "mse" : 0, "psnr": 0}

        result_overlap = cv2.copyTo(composition_img_cp, overlap_mask)[o_y:o_y+o_h, o_x:o_x+o_w]
        src_overlap = cv2.copyTo(src_img_cp, overlap_mask)[o_y:o_y+o_h, o_x:o_x+o_w]
        dst_overlap = cv2.copyTo(dst_img_cp, overlap_mask)[o_y:o_y+o_h, o_x:o_x+o_w]

        ssim_score = (self.__calculate_ssim(dst_overlap, result_overlap) + self.__calculate_ssim(src_overlap, result_overlap)) / 2
        mse = (self.__calculate_mse(dst_overlap, result_overlap) + self.__calculate_mse(src_overlap, result_overlap)) / 2
        psnr = (self.__calculate_psnr(dst_overlap, result_overlap) + self.__calculate_psnr(src_overlap, result_overlap)) / 2
        seam_score = self.__calculate_seam_score(result_overlap)

        return {"ssim": ssim_score, "mse" : mse, "psnr": psnr, "seam_score": seam_score}

    
    def get_warp_metrics(self, src_img: Mat, dst_img: Mat, src_mask: Mat, dst_mask: Mat) -> dict[str, float]:
        overlap_mask = cv2.bitwise_and(src_mask, dst_mask).astype(np.uint8)
        src_img_cp = src_img.astype(np.uint8)
        dst_img_cp = dst_img.astype(np.uint8)

        contours, _ = cv2.findContours(cv2.cvtColor(overlap_mask, cv2.COLOR_RGB2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return {"ssim": 0, "mse" : 0, "psnr": 0}

        mainContour = contours[0]
        o_x, o_y, o_w, o_h = cv2.boundingRect(mainContour)

        if o_w < 7 or o_h < 7:
            return {"ssim": 0, "mse" : 0, "psnr": 0}

        src_overlap = cv2.copyTo(src_img_cp, overlap_mask)[o_y:o_y+o_h, o_x:o_x+o_w]
        dst_overlap = cv2.copyTo(dst_img_cp, overlap_mask)[o_y:o_y+o_h, o_x:o_x+o_w]

        ssim_score = self.__calculate_ssim(src_overlap, dst_overlap)
        mse = self.__calculate_mse(src_overlap, dst_overlap)
        psnr = self.__calculate_psnr(src_overlap, dst_overlap)

        return {"ssim": ssim_score, "mse" : mse, "psnr": psnr}
    
    def get_matching_metrics(self, keypoints1: np.array, keypoints2: np.array, auc_thresholds: np.array) -> dict[str, float]:
        if(len(keypoints1) < 4):
            return {"auc": 0}

        keypoints1_cp = np.array(keypoints1)
        keypoints2_cp = np.array(keypoints2)

        H, _ = cv2.findHomography(keypoints2_cp, keypoints1_cp, cv2.RANSAC)

        pts2_proj = cv2.perspectiveTransform(keypoints2_cp.reshape(-1, 1, 2), H).reshape(-1, 2)
        reprojection_error = np.linalg.norm(keypoints1_cp - pts2_proj, axis=1)

        inliers = [np.mean(reprojection_error <= t) for t in auc_thresholds]
        area_under_curve = metrics.auc(auc_thresholds, inliers)
        return {"auc": area_under_curve}