import os
import random
import shutil
from collections import defaultdict
from tqdm import tqdm

from utils.preprocessing import verify_split
import config

def splitter(images_dir, masks_dir, train_images_dir, train_masks_dir, valid_images_dir, valid_masks_dir, verification=True, split_ratio=0.7):

    for folder in [
        train_images_dir, train_masks_dir,
        valid_images_dir, valid_masks_dir
    ]:
        os.makedirs(folder, exist_ok=True)

    groups = defaultdict(list)

    for filename in os.listdir(images_dir):
        if not filename.endswith('.png'):
            continue
        if not os.path.isfile(os.path.join(masks_dir, filename)):
            continue

        render_id = filename.split("frame")[0]
        groups[render_id].append(filename)

    train_files = []
    valid_files = []



    for render_id, files in groups.items():
        files.sort()
        random.shuffle(files)
        split_index = int(len(files) * split_ratio)
        train_files.extend(files[:split_index])
        valid_files.extend(files[split_index:])


    for f in tqdm(train_files, desc="Copying train files"):
        shutil.copy(os.path.join(images_dir,f), train_images_dir)
        shutil.copy(os.path.join(masks_dir,f), train_masks_dir)

    for f in tqdm(valid_files, desc="Copying valid files"):
        shutil.copy(os.path.join(images_dir,f), valid_images_dir)
        shutil.copy(os.path.join(masks_dir,f), valid_masks_dir)

    print("Split completed")
    print(f"Train: {len(train_files)}")
    print(f"Valid: {len(valid_files)}")

    if verification == True:
        verify_split(train_images_dir, train_masks_dir)
        verify_split(valid_images_dir, valid_masks_dir)

    return 0


if __name__ == "__main__":

    splitter(config.PREPROCESSED_IMAGES_DIR, config.PREPROCESSED_MASKS_DIR,
            config.DATASET_IMAGES_TRAIN_DIR, config.DATASET_MASKS_TRAIN_DIR,
            config.DATASET_IMAGES_VALID_DIR, config.DATASET_MASKS_VALID_DIR)
