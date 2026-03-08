import os 
import random
import shutil
from collections import defaultdict

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

images_dir = os.path.join(base_dir, "camera_pose/datasetProcessedSynthetic/images")
masks_dir = os.path.join(base_dir, "camera_pose/datasetProcessedSynthetic/masks")

output_train_images = os.path.join(base_dir, "camera_pose/datasetStructuredSynthetic/train/images")
output_train_masks = os.path.join(base_dir, "camera_pose/datasetStructuredSynthetic/train/masks")
output_valid_images = os.path.join(base_dir, "camera_pose/datasetStructuredSynthetic/valid/images")
output_valid_masks = os.path.join(base_dir, "camera_pose/datasetStructuredSynthetic/valid/masks")

split_ratio = 0.7
seed = 42

random.seed(seed)

for folder in [
    output_train_images, output_train_masks,
    output_valid_images, output_valid_masks
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


for f in train_files:
    shutil.copy(os.path.join(images_dir,f), output_train_images)
    shutil.copy(os.path.join(masks_dir,f), output_train_masks)

for f in valid_files:
    shutil.copy(os.path.join(images_dir,f), output_valid_images)
    shutil.copy(os.path.join(masks_dir,f), output_valid_masks)

print("Split completado")
print(f"Train: {len(train_files)}")
print(f"Valid: {len(valid_files)}")



def verify_split(images_folder, masks_folder):
    image_files = sorted(os.listdir(images_folder))
    mask_fles = sorted(os.listdir(masks_folder))

    image_set = set(image_files)
    mask_set = set(mask_fles)

    missing_masks = image_set - mask_set
    missing_images = mask_set - image_set

    if len(missing_masks) == 0 and len(missing_images) == 0:
        print(f"Correct Verification in {images_folder}")
        print(f"Total files: {len(image_files)}")
    else:
        print("Problem detected")
        if missing_masks:
            print("Images without mask:", missing_masks)
        if missing_images:
            print("Masks without image:", missing_images)

verify_split(output_train_images, output_train_masks)
verify_split(output_valid_images, output_valid_masks)
