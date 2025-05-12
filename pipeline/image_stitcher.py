# @Author: Pavlo Butenko

from cv2 import Mat
import numpy as np

from pipeline.enums import ComposerType, EnvironmentType, MatcherType
from pipeline.Modules.match_finders import FeatureDetector
from pipeline.Modules.warp_composer import DetectorFreeModel, DetectorFreeWarper, FeatureDetectorWarper, UDIS2Warper, Warper
from pipeline.Modules.composition_module import AlphaCompositionModule, WeightedAlphaCompositionModule, CompositionModule, SimpleCompositionModule, UdisCompositionModule

class ImageStitcher:
    """
    Configurable image stitcher class.
    This class allows to configure the image stitcher with different feature matching and composition strategies.
    """
    warper: Warper
    composer: CompositionModule
    def __init__(self, matcher_type: MatcherType, composer_type: ComposerType, weights_path: str | None = None, environment: EnvironmentType | None = None):
        """
        Initialize the ImageStitcher with the specified feature matcher and composer types.
        
        Arguments:
            matcher_type: The type of feature matcher to use (SIFT, LoFTR, AdaMatcher).
            composer_type: The composition strategy to use (Simple, SimpleAlpha, WeightedAlpha, UDIS2).
            weights_path: Path to the weights file for the feature matcher model, if not specified default path is used.
            environment: Environment type for the matcher (Applicable to LoFTR).
        """
        if (matcher_type == MatcherType.ORB or matcher_type == MatcherType.AKAZE
            or matcher_type == MatcherType.SIFT or matcher_type == MatcherType.BRISK):
            self.warper = FeatureDetectorWarper(FeatureDetector[matcher_type.name])
        elif matcher_type == MatcherType.LoFTR or matcher_type == MatcherType.AdaMatcher:
            if matcher_type == MatcherType.LoFTR and environment == None:
                raise Exception("Environment type must be provided")
            self.warper = DetectorFreeWarper(DetectorFreeModel[matcher_type.name], environment, weights_path)
        elif matcher_type == MatcherType.UDIS2:
            self.warper = UDIS2Warper()
        else:
            raise Exception("Not supported")
        
        if composer_type == ComposerType.Overlay:
            self.composer = SimpleCompositionModule()
        elif composer_type == ComposerType.SimpleAlpha:
            self.composer = AlphaCompositionModule()
        elif composer_type == ComposerType.WeightedAlpha:
            self.composer = WeightedAlphaCompositionModule()
        elif composer_type == ComposerType.UDIS2:
            self.composer = UdisCompositionModule()
        else:
            raise Exception("Not supported")

    def stitch(self, img1: Mat, img2: Mat) -> tuple[Mat, Mat, Mat] | None:
        """
        Stitch two images together using the configured warper and composer.
        
        Arguments:
            img1: The first image to stitch.
            img2: The second image to stitch.
        
        Returns:
            A tuple containing the stitched image, individual images' masks indicating their influence 
            over the resulting image.
            If stitching fails, returns None.
        """
        warp_result = self.warper.construct_warp(img1, img2)

        if warp_result == None:
            return None
        warp1, warp2, mask1, mask2 = warp_result

        composition_result = self.composer.composite(warp1, mask1, warp2, mask2)
        
        if composition_result == None:
            return None

        result, result_mask1, result_mask2 = composition_result

        return result.astype(dtype=np.uint8), result_mask1.astype(dtype=np.uint8), result_mask2.astype(dtype=np.uint8)