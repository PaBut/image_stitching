from enum import Enum
import time

from cv2 import Mat

from pipeline.enums import EnvironmentType
from Modules.match_finders import FeatureDetector
from Modules.warp_composer import DetectorFreeModel, DetectorFreeWarper, FeatureDetectorWarper, UDIS2Warper, Warper
from Modules.composition_module import AlphaCompositionModule, ComplexAlphaCompositionModule, CompositionModule, SimpleCompositionModule, UdisCompositionModule

class DetectorType(Enum):
    ORB = 0
    SIFT = 1
    BRISK = 2
    AKAZE = 3
    LoFTR = 4
    AdaMatcher = 5
    UDIS2 = 6

class ComposerType(Enum):
    Simple = 0
    SimpleAlpha = 1
    ComplexAlpha = 2
    UDIS2 = 3

class ImageStitcher:
    warper: Warper
    composer: CompositionModule
    def __init__(self, detector_type: DetectorType, composer_type: ComposerType, environment: EnvironmentType | None = None):

        if (detector_type == DetectorType.ORB or detector_type == DetectorType.AKAZE
            or detector_type == DetectorType.SIFT or detector_type == DetectorType.BRISK):
            self.warper = FeatureDetectorWarper(FeatureDetector[detector_type.name])
        elif detector_type == DetectorType.LoFTR or detector_type == DetectorType.AdaMatcher:
            if detector_type == DetectorType.LoFTR and environment == None:
                raise Exception("Environment type msut be provided")
            self.warper = DetectorFreeWarper(DetectorFreeModel[detector_type.name], environment)
        elif detector_type == DetectorType.UDIS2:
            self.warper = UDIS2Warper()
        else:
            raise Exception("Not supported")
        
        if composer_type == ComposerType.Simple:
            self.composer = SimpleCompositionModule()
        elif composer_type == ComposerType.SimpleAlpha:
            self.composer = AlphaCompositionModule()
        elif composer_type == ComposerType.ComplexAlpha:
            self.composer = ComplexAlphaCompositionModule()
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