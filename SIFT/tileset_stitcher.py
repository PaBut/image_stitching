from cv2 import Mat
import cv2
import numpy as np
from composition_module import AlphaCompositionModule, ComplexAlphaCompositionModule, SimpleCompositionModule, UdisCompositionModule
from enums import EnvironmentType
from image_stitcher import ComposerType, DetectorType
from match_finders import AdaMatcherMatchFinder, FeatureDetector, FeatureDetectorMatchFinder, LoFTRMatchFinder
from tiles_loader import TilesLoader
from warp_composer import DetectorFreeModel, DetectorFreeWarper, FeatureDetectorWarper, UDIS2Warper


class TilesetStitcher:
    def __init__(self, detector_type: DetectorType, composer_type: ComposerType, environment: EnvironmentType):
        if (detector_type == DetectorType.ORB or detector_type == DetectorType.AKAZE
            or detector_type == DetectorType.SIFT or detector_type == DetectorType.BRISK):
            self.match_finder = FeatureDetectorMatchFinder(detector_type)
        elif detector_type == DetectorType.LoFTR:
            if environment == None:
                raise Exception("Environment type msut be provided")
            self.match_finder = LoFTRMatchFinder(environment)
        elif detector_type == DetectorType.AdaMatcher:
            self.match_finder = AdaMatcherMatchFinder()
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
    def stitch_tileset(self, tiles_loader: TilesLoader):
        width = tiles_loader.width()
        height = tiles_loader.height()
        tile_height, tile_width = tiles_loader.get_tile(0, 0).shape[:2]
        homographies: dict[tuple[tuple[int, int], tuple[int, int]], Mat] = {}
        for x in range(width):
            for y in range(height):
                if x + 1 < width:
                    tile1 = tiles_loader.get_tile(x, y)
                    tile2 = tiles_loader.get_tile(x + 1, y)
                    keypoints1, keypoints2 = self.match_finder.find_matches(tile1, tile2)
                    keypoints1_copy = np.copy(keypoints1)
                    keypoints1 = keypoints1[(keypoints1[:, 0] > 0.95 * tile_width) & (keypoints2[:, 0] < 0.05 * tile_width)]
                    keypoints2 = keypoints2[(keypoints1_copy[:, 0] > 0.95 * tile_width) & (keypoints2[:, 0] < 0.05 * tile_width)]
                    H, _ = cv2.findHomography(np.array(keypoints2), np.array(keypoints1), cv2.RANSAC, 5.0)
                    homographies[((x, y), (x + 1, y))] = H
                if y + 1 < height: 
                    tile1 = tiles_loader.get_tile(x, y)
                    tile2 = tiles_loader.get_tile(x, y + 1)
                    keypoints1, keypoints2 = self.match_finder.find_matches(tile1, tile2)
                    keypoints1_copy = np.copy(keypoints1)
                    keypoints1 = keypoints1[(keypoints1[0, :] > 0.95 * tile_height) & (keypoints2[0, :] < 0.05 * tile_height)]
                    keypoints2 = keypoints2[(keypoints1_copy[0, :] > 0.95 * tile_height) & (keypoints2[0, :] < 0.05 * tile_width)]
                    H, _ = cv2.findHomography(np.array(keypoints2), np.array(keypoints1), cv2.RANSAC, 5.0)
                    homographies[((x, y), (x, y + 1))] = H

        

        