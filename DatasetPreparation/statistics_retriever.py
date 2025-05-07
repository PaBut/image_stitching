import os
import sys
import numpy as np

def process_file(filename, base_path):
    img_count = 0
    pair_count = 0
    valid_pair_count = 0

    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                npz_path = os.path.join(base_path, line.strip()) + ".npz"
                if not os.path.exists(npz_path):
                    print(f"File not found: {npz_path}")
                    continue
                npz = np.load(npz_path, allow_pickle=True)
                images = npz['image_paths']
                images = images[images != None]
                pairs = npz['pair_infos']

                img_count += len(images)
                pair_count += len(pairs)
                valid_pair_count += len([pair for pair in pairs if pair[1] > 0.15])

                print(f"Processing {npz_path}:")
                print(f"  Image count: {len(images)}")
                print(f"  Pair count: {len(pairs)}")
                print(f"  Valid pair count: {len([pair for pair in pairs if pair[1] > 0.15])}")
    except Exception as e:
        print(f"An error occurred: {e}")

    print(f"Image count: {img_count}")
    print(f"Pair count: {pair_count}")
    print(f"Valid pair count: {valid_pair_count}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <filename.txt> <base_path>")
    else:
        process_file(sys.argv[1], sys.argv[2])