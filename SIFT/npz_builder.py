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
    poses = {}
    camera_intristics = {}

    with open(images_txt_path, 'r') as f:
        lines = f.readlines()
        for i in range(4, len(lines), 2):  # Every image has 2 lines
            parts = lines[i].split()
            image_id = int(parts[0])
            image_name = parts[9]
            if prefix is None or image_name.startswith(prefix):
                camera_id = int(parts[8])
                qw, qx, qy, qz = map(float, parts[1:5])  # Quaternion rotation
                tx, ty, tz = map(float, parts[5:8])      # Translation vector
                rotation_matrix = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()  # Convert to matrix
                poses[image_id] = np.hstack([rotation_matrix, np.array([tx, ty, tz]).reshape(-1, 1)])
                image_paths[image_id] = os.path.join(IMAGE_PATH, image_name)
                camera_intristics[image_id] = camera_info[camera_id]

    return image_paths, poses, camera_intristics

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
            
            camera_data[camera_id] = construct_intrinsic_matrix(fx, fy, cx, cy)
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
                    if image_id not in image_points:
                        image_points[image_id] = []
                    image_points[image_id].append(point_id)
    
    return image_points

def compute_overlap(image_points, image_names):
    """Computes overlap coefficient for each image pair"""
    image_ids = list(image_points.keys())
    overlap_scores = {}

    for i in range(len(image_ids)):
        for j in range(i + 1, len(image_ids)):
            img1, img2 = image_ids[i], image_ids[j]
            shared_points = len(image_points[img1] & image_points[img2])
            min_points = min(len(image_points[img1]), len(image_points[img2]))
            
            if min_points > 0:  # Avoid division by zero
                overlap = shared_points / min_points
                overlap_scores.append(((image_names[img1], image_names[img2]), overlap, []))

    return overlap_scores

def get_depthmap_paths(image_paths):
    """Generates full image paths based on base path and prefix"""
    depth_maps = map(lambda x: x.replace(IMAGE_PATH, DEPTHMAP_PATH), image_names.copy())
    return image_paths

def parse_args():
    """Parse command line arguments"""
    parser = ArgumentParser(description="Process COLMAP data")
    parser.add_argument('--scene_name', type=str, required=True, help='Scene name')
    parser.add_argument('--image_prefix', type=str, required=True, help='Image prefix')
    parser.add_argument('--cameras_path', type=str, required=True, help='Cameras file path')
    parser.add_argument('--images_path', type=str, required=True, help='Images file path')
    parser.add_argument('--points3D_path', type=str, required=False, default=None, help='Points3D file path')
    return parser.parse_args()

if __name__ == "__main__":    
    args = parse_args()

    scene_name = args.scene_name
    images_txt_path = args.images_path
    points3D_txt_path = args.points3D_path
    camera_txt_path = args.cameras_path
    prefix = args.image_prefix

    cameras = load_camera_txt(camera_txt_path)
    print(f"Loaded {len(cameras)} cameras")
    image_names, poses, camera_intristics = load_images_txt(images_txt_path, cameras, prefix)
    print(f"Loaded {len(image_names)} images")
    depthmaps = get_depthmap_paths(image_names)
    image_points = load_points3D_txt(points3D_txt_path, [image for image
                                                        in image_names.keys() if image_names[image] != None])
    overlap_results = compute_overlap(image_points, image_names)
    print(f"Computed overlap coefficients for {len(overlap_results)} pairs")

    np.savez(f"{scene_name}.npz", image_paths=image_names, depth_paths=depthmaps,
            poses=poses, camera_intristics=camera_intristics, pair_infos=overlap_results)

    print(f"Saved overlap coefficients to {scene_name}.npz")
