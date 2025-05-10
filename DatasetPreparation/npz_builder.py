# @Author: Pavlo Butenko

from collections import defaultdict
import os
import numpy as np
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation

IMAGE_PATH = "images/"
DEPTHMAP_PATH = "depth_maps/"

def load_images_txt(images_txt_path, camera_info, prefix=None):
    """Loads image poses and paths from COLMAP images.txt"""
    image_paths = {}
    depth_paths = {}
    poses = {}
    camera_intristics = {}
    image_points = {}

    with open(images_txt_path, 'r') as f:
        lines = f.readlines()
        for i in range(4, len(lines), 2):  # Every image has 2 lines
            parts = lines[i].split()
            image_id = int(parts[0])
            image_name = parts[9]
            if prefix is None or image_name.startswith(prefix):
                camera_id = int(parts[8])

                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])    
                rotation_matrix = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                
                keypoints_data = lines[i+1].split()
                point3d_ids = [int(keypoints_data[j]) for j in range(2, len(keypoints_data), 3) if int(keypoints_data[j]) != -1]
                image_points[image_id] = set(point3d_ids)
                
                pose_matrix = np.eye(4) 
                pose_matrix[:3, :3] = rotation_matrix
                pose_matrix[:3, 3] = [tx, ty, tz]

                poses[image_id] = pose_matrix
                image_paths[image_id] = os.path.join(IMAGE_PATH, image_name)
                depth_paths[image_id] = os.path.join(DEPTHMAP_PATH, image_name) + ".photometric.bin"
                camera_intristics[image_id] = camera_info[camera_id]

    return image_paths, depth_paths, poses, camera_intristics, image_points

def construct_intrinsic_matrix(fx, fy, cx, cy):
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    return K

def load_camera_txt(camera_txt_path):
    """Loads camera intrinsics from COLMAP cameras.txt"""
    camera_data = {}
    with open(camera_txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines[3:]:  # Skip first 3 header lines
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            params = list(map(float, parts[4:]))
            
            if model == "SIMPLE_PINHOLE":
                fx = params[0]
                fy = params[0]
                cx = params[1]
                cy = params[2]
            elif model == "PINHOLE" or model == "SIMPLE_RADIAL":
                fx = params[0]
                fy = params[1]
                cx = params[2]
                cy = params[3]
            else:
                raise ValueError(f"Unsupported camera model: {model}")
            
            camera_data[camera_id] = (construct_intrinsic_matrix(fx, fy, cx, cy), int(parts[2]), int(parts[3]))  # K, width, height
    return camera_data

def load_points3D_txt(points3D_txt_path, available_images=None):
    """Loads 3D point observations from COLMAP points3D.txt"""
    image_points = defaultdict(set)
    with open(points3D_txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines[3:]:  # Skip first 3 header lines
            parts = line.split()
            if len(parts) < 8:
                continue  # Skip invalid lines
            
            point_id = int(parts[0])
            track = parts[8:]  # The track starts from the 9th column
            
            for j in range(0, len(track), 2):
                image_id = int(track[j])
                if image_id in available_images:
                    image_points[image_id].add(point_id)
    
    return image_points

def compute_overlap(image_points):
    """Computes overlap coefficient for each image pair"""
    image_ids = list(image_points.keys())
    overlap_scores = []

    for i in range(len(image_ids)):
        for j in range(i + 1, len(image_ids)):
            img1, img2 = image_ids[i], image_ids[j]

            common_points = len(image_points[img1] & image_points[img2])
            min_points = min(len(image_points[img1]), len(image_points[img2]))

            score = common_points / min_points if min_points > 0 else 0
            overlap_scores.append(((img1, img2), score, []))

    return overlap_scores

def convert_dict_tondarray(dict_data):
    max_index = max(dict_data.keys()) + 1

    array = [None] * max_index

    for key, value in dict_data.items():
        array[key] = value

    return np.array(array, dtype=object)

def parse_args():
    """Parse command line arguments"""
    parser = ArgumentParser(description="Process COLMAP data")
    parser.add_argument('--output', type=str, required=True, help='Output path')
    parser.add_argument('--image_prefix', type=str, required=False, help='Image prefix')
    parser.add_argument('--cameras_path', type=str, required=True, help='Cameras file path')
    parser.add_argument('--images_path', type=str, required=True, help='Images file path')
    parser.add_argument('--points3D_path', type=str, required=True, default=None, help='Points3D file path')
    return parser.parse_args()

if __name__ == "__main__":    
    args = parse_args()

    output_path = args.output
    images_txt_path = args.images_path
    points3D_txt_path = args.points3D_path
    camera_txt_path = args.cameras_path
    prefix = args.image_prefix

    cameras = load_camera_txt(camera_txt_path)
    print(f"Loaded {len(cameras)} cameras")
    image_names, depthmaps, poses, camera_intristics, image_points = load_images_txt(images_txt_path, cameras, prefix)
    print(f"Loaded {len(image_names)} images")
    image_points = load_points3D_txt(points3D_txt_path, [image for image
                                                        in image_names.keys() if image_names[image] != None])
    overlap_results = compute_overlap(image_points)
    print(f"Computed overlap coefficients for {len(overlap_results)} pairs")

    np.savez(output_path, allow_pickle=True, image_paths=convert_dict_tondarray(image_names), depth_paths=convert_dict_tondarray(depthmaps),
            poses=convert_dict_tondarray(poses), intrinsics=convert_dict_tondarray({key: value[0] for key, value in camera_intristics.items()}),
            pair_infos=np.array(overlap_results, dtype=np.dtype([
                ('image_pair', 'i4', (2,)),
                ('overlap', 'f4'),       
                ('extra_data', 'O')
            ])))

    print(f"Saved overlap coefficients to {output_path}")
