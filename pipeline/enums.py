# @Author: Pavlo Butenko

from enum import Enum

class MatcherType(Enum):
    """Feature matcher types"""
    ORB = 0
    SIFT = 1
    BRISK = 2
    AKAZE = 3
    LoFTR = 4
    AdaMatcher = 5
    UDIS2 = 6

class ComposerType(Enum):
    """Composition strategies"""
    Simple = 0
    SimpleAlpha = 1
    WeightedAlpha = 2
    UDIS2 = 3

class EnvironmentType(Enum):
    """Environment type for LoFTR network"""
    Indoor = 0
    Outdoor = 1