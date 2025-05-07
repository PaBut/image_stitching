import cv2
from Pipeline.common import enum_from_string
from Pipeline.enums import EnvironmentType
from Pipeline.image_stitcher import ComposerType, DetectorType, ImageStitcher
import argparse

def case_insensitive_choices(choices):
    def validate(value):
        value_lower = value.lower()
        if value_lower not in choices:
            raise argparse.ArgumentTypeError(f"Invalid choice: '{value}'. Choose from {choices}.")
        return value_lower
    return validate

def parse_args():
    feature_finder_methods = [method.name.lower() for method in DetectorType]
    composition_methods = [method.name.lower() for method in ComposerType]
    environment_types = [type.name.lower() for type in EnvironmentType]
    parser = argparse.ArgumentParser("Image stitcher with different methods")
    parser.add_argument('img1_path', type=str, help="Path to the first image")
    parser.add_argument('img2_path', type=str, help="Path to the second image")
    parser.add_argument('result_path', type=str, help="Path to the resulting image")
    parser.add_argument('--mfinder', help="Feature finder method", type=case_insensitive_choices(feature_finder_methods), required=True)
    parser.add_argument('--composition', help="Image composition method", type=case_insensitive_choices(composition_methods), required=True)
    parser.add_argument('--environment', help="Environment type:(indoor or outdoor)", type=case_insensitive_choices(environment_types), default=EnvironmentType.Indoor.name)

    return parser.parse_args()


args = parse_args()

detector_type = enum_from_string(args.mfinder, DetectorType)
composition_type = enum_from_string(args.composition, ComposerType)
environment_type = enum_from_string(args.environment, EnvironmentType)

img1 = cv2.imread(args.img1_path)
img2 = cv2.imread(args.img2_path)

result_path = args.result_path

stitcher = ImageStitcher(detector_type, composition_type, environment_type)

result = stitcher.stitch(img1, img2)

if result == None:
    print("Stitching process failed")
    exit(1)
result_img, _, _ = result
cv2.imwrite(result_path, result_img)