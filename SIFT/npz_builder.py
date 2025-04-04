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
    image_points = {}

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
                keypoints_data = lines[i+1].split()
                point3d_ids = [int(keypoints_data[j]) for j in range(2, len(keypoints_data), 3) if int(keypoints_data[j]) != -1]
                image_points[image_id] = set(point3d_ids)
                poses[image_id] = np.hstack([rotation_matrix, np.array([tx, ty, tz]).reshape(-1, 1)])
                image_paths[image_id] = os.path.join(IMAGE_PATH, image_name)
                camera_intristics[image_id] = camera_info[camera_id]

    return image_paths, poses, camera_intristics, image_points

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
            points1 = image_points[img1]
            points2 = image_points[img2]
            # common_points = len(points1.intersection(points2))
            # total_points = len(points1.union(points2))
            common_points = len(image_points[img1] & image_points[img2])
            min_points = min(len(image_points[img1]), len(image_points[img2]))
            score = common_points / min_points if min_points > 0 else 0
            overlap_scores.append(((img1, img2), score, []))

    return overlap_scores
 
############################

# def angle_between_vectors(v1, v2):
#     """Compute angle (in degrees) between two vectors."""
#     v1 = v1 / np.linalg.norm(v1)
#     v2 = v2 / np.linalg.norm(v2)
#     dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
#     return np.arccos(dot_product) * 180 / np.pi

# def compute_fov(intrinsics):
#     """Compute horizontal and vertical FOV in degrees."""
#     fx, fy, width, height = intrinsics[0][0][0], intrinsics[0][1][1], intrinsics[1], intrinsics[2]
#     fov_h = 2 * np.arctan(width / (2 * fx)) * 180 / np.pi
#     fov_v = 2 * np.arctan(height / (2 * fy)) * 180 / np.pi
#     return fov_h, fov_v

# def extract_from_pose_matrix(pose_matrix):
#     """Extract rotation (R) and translation (t) from a 4x4 pose matrix."""
#     R = pose_matrix[0:3, 0:3]  # Top-left 3x3
#     t = pose_matrix[0:3, 3]    # Top-right 3x1
#     return R, t

# def estimate_overlap(images, cameras, poses):
#     """Estimate overlapping ratio using poses and intrinsics."""
#     pairs = []
#     image_ids = list(images.keys())
    
#     for i in range(len(image_ids)):
#         for j in range(i+1, len(image_ids)):
#             id1, id2 = image_ids[i], image_ids[j]

#             R1, t1 = extract_from_pose_matrix(poses[id1])
#             R2, t2 = extract_from_pose_matrix(poses[id2])
            
#             # Camera centers: C = -R^T * t
#             C1 = -R1.T.dot(t1)
#             C2 = -R2.T.dot(t2)
#             distance = np.linalg.norm(C1 - C2)
            
#             # Optical axes
#             z_axis = np.array([0, 0, 1])
#             dir1 = R1.dot(z_axis)
#             dir2 = R2.dot(z_axis)
#             angle = angle_between_vectors(dir1, dir2)
            
#             # FOV from intrinsics
#             fov_h1, fov_v1 = compute_fov(cameras[id1])
#             fov_h2, fov_v2 = compute_fov(cameras[id2])
#             avg_fov = (fov_h1 + fov_h2) / 2  # Use horizontal FOV for simplicity
            
#             # Simple overlap heuristic
#             # Vector from C1 to C2
#             # C1_to_C2 = C2 - C1
#             # angle_to_C2 = angle_between_vectors(dir1, C1_to_C2)
            
#             # Check if C2 is within img1's FOV (and vice versa)
#                 # Overlap ratio: decreases with angle and distance
#             overlap = (1 - angle / avg_fov) * (1 - distance / 10.0)
#             pairs.append((
#                 (id1, id2), max(0, overlap), []  # Ensure non-negative
#             ))
    
#     return pairs

##############################

def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % 2147483647  # Lower 31 bits
    image_id1 = pair_id // 2147483647  # Upper 31 bits
    return image_id1, image_id2

def get_pairs(db_path, image_idx):
    """Get image pairs and their match counts from the database"""
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all matches
    cursor.execute("SELECT pair_id FROM matches")
    matches = cursor.fetchall()

    # Close the connection
    conn.close()

    # Extract and print image pairs
    image_pairs = []
    for pair_id in matches:
        image_id1, image_id2 = pair_id_to_image_ids(pair_id[0])
        if image_id1 in image_idx and image_id2 in image_idx:
            pair = ((image_id1, image_id2), 1, [])
            image_pairs.append(pair)
    
    return image_pairs

def get_depthmap_paths(image_paths):
    """Generates full image paths based on base path and prefix"""
    depth_maps = map(lambda x: x.replace(IMAGE_PATH, DEPTHMAP_PATH), image_names.copy())
    return image_paths

def convert_dict_tondarray(dict_data):
    max_index = max(dict_data.keys()) + 1  # Ensure space for the largest index

    # Create an empty array (or fill with a default value like -1)
    array = [None] * max_index  # Shape: (max_index, 3)

    # Fill the array using the dictionary keys as indices
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
    # parser.add_argument('--db_path', type=str, required=True, default=None, help='Database file path')
    return parser.parse_args()

if __name__ == "__main__":    
    args = parse_args()

    output_path = args.output
    images_txt_path = args.images_path
    points3D_txt_path = args.points3D_path
    camera_txt_path = args.cameras_path
    prefix = args.image_prefix
    # db_path = args.db_path

    cameras = load_camera_txt(camera_txt_path)
    print(f"Loaded {len(cameras)} cameras")
    image_names, poses, camera_intristics, image_points = load_images_txt(images_txt_path, cameras, prefix)
    print(f"Loaded {len(image_names)} images")
    depthmaps = get_depthmap_paths(image_names)
    image_points = load_points3D_txt(points3D_txt_path, [image for image
                                                        in image_names.keys() if image_names[image] != None])
    # overlap_results = get_pairs(db_path, image_names.keys())
    overlap_results = compute_overlap(image_points)
    print(f"Computed overlap coefficients for {len(overlap_results)} pairs")

    np.savez(output_path, image_paths=convert_dict_tondarray(image_names), depth_paths=convert_dict_tondarray(depthmaps),
            poses=convert_dict_tondarray(poses), camera_intristics=convert_dict_tondarray({key: value[0] for key, value in camera_intristics.items()}),
            pair_infos=np.array(overlap_results, dtype=np.dtype([
                ('image_pair', 'i4', (2,)),  # Pair of int IDs
                ('overlap', 'f4'),           # Overlap coefficient (float)
                ('extra_data', 'O')          # List of floats (object type)
            ])))

    print(f"Saved overlap coefficients to {output_path}")
