from enum import Enum

from cv2 import Mat

from pipeline.enums import ComposerType, EnvironmentType, MatcherType
from pipeline.Modules.match_finders import FeatureDetector
from pipeline.Modules.warp_composer import DetectorFreeModel, DetectorFreeWarper, FeatureDetectorWarper, UDIS2Warper, Warper
from pipeline.Modules.composition_module import AlphaCompositionModule, WeightedAlphaCompositionModule, CompositionModule, SimpleCompositionModule, UdisCompositionModule

class ImageStitcher:
    warper: Warper
    composer: CompositionModule
    def __init__(self, matcher_type: MatcherType, composer_type: ComposerType, weights_path: str | None = None, environment: EnvironmentType | None = None):

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
        
        if composer_type == ComposerType.Simple:
            self.composer = SimpleCompositionModule()
        elif composer_type == ComposerType.SimpleAlpha:
            self.composer = AlphaCompositionModule()
        elif composer_type == ComposerType.ComplexAlpha:
            self.composer = WeightedAlphaCompositionModule()
        elif composer_type == ComposerType.UDIS2:
            self.composer = UdisCompositionModule()
        else:
            raise Exception("Not supported")

    def stitch(self, img1: Mat, img2: Mat):
        warp_result = self.warper.construct_warp(img1, img2)

        if warp_result == None:
            return None
        warp1, warp2, mask1, mask2 = warp_result

        result, result_mask1, result_mask2 = self.composer.composite(warp1, mask1, warp2, mask2)

        return result, result_mask1, result_mask2