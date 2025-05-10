# @Author: Pavlo Butenko

from enum import Enum

class MatcherType(Enum):
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
    WeightedAlpha = 2
    UDIS2 = 3

class EnvironmentType(Enum):
    Indoor = 0
    Outdoor = 1