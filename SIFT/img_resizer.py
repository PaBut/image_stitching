import os
import sys
from PIL import Image


def resize_image(input_path, output_path, size):
    """Resizes the image and saves it to the output path."""
    with Image.open(input_path) as img:
        # Resize the image
        img = img.resize(size)
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Save the resized image
        img.save(output_path)
        print(f"Saved resized image to {output_path}")

def process_images(input_dir, output_dir, size):
    """Recursively processes all images in the input directory."""
    for root, dirs, files in os.walk(input_dir):
        print(files)
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                input_path = os.path.join(root, file)
                
                # Create the relative path for the output
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                
                # Resize the image and save it
                resize_image(input_path, output_path, size)

if len(sys.argv) < 3:
    exit(1)

# Input and output directories
input_directory = sys.argv[1]  # Folder with the original images
output_directory = sys.argv[2]  # Folder to save resized images

# Set the desired size (width, height)
new_size = (1008, 567)

# Start processing images
process_images(input_directory, output_directory, new_size)
