import os
import argparse
from glob import glob
import random
import shutil
from tqdm import tqdm

def main(images_dir, portion, output_dir):
    images = glob(os.path.join(images_dir, "*.jpg"))
    random.seed(0)
    random.shuffle(images)
    picked_images = images[:int(len(images) * portion)]
    os.makedirs(output_dir, exist_ok=True)
    for image in tqdm(picked_images):
        shutil.copy(image, os.path.join(output_dir, os.path.basename(image)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--portion", default=0.01, type=float, required=False)
    parser.add_argument("--output", type=str, required=True)
    
    args = parser.parse_args()

    main(args.images, args.portion, args.output)