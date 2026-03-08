from PIL import Image
import numpy as np
import os
from tqdm import tqdm


from concurrent.futures import ProcessPoolExecutor
from camera_pose.utils.palette import exactColorPalette, rangeColorPalette


def dataProcess(args):

    inputMaskPath, outputMaskPath, inputImagePath, outputImagePath = args
    mask = Image.open(inputMaskPath).convert("RGB")
    mask_pixels = np.array(mask)
    height, width = mask_pixels.shape[:2]
    grayMask = np.zeros((height, width), dtype=np.uint8)

    r_channel = mask_pixels[:, :, 0]
    g_channel = mask_pixels[:, :, 1]
    b_channel = mask_pixels[:, :, 2]

    # Exact color matches
    for color, classIndex in exactColorPalette.items():
        match = np.all(mask_pixels == color, axis=-1)  # shape (H, W)
        grayMask[match] = classIndex

    # Range-based matches
    for classInfo in rangeColorPalette.values():
        (r_min, r_max), (g_min, g_max), (b_min, b_max) = classInfo["range"]
        classIndex = classInfo["classIndex"]
        match = (
            (r_channel >= r_min) & (r_channel <= r_max) &
            (g_channel >= g_min) & (g_channel <= g_max) &
            (b_channel >= b_min) & (b_channel <= b_max)
        )
        grayMask[match] = classIndex
    grayMaskImage = Image.fromarray(grayMask)
    grayMaskImageResized = grayMaskImage.resize((512,512), resample=Image.NEAREST)
    grayMaskImageResized.save(outputMaskPath)

    image = Image.open(inputImagePath).convert("RGB")
    imageResized = image.resize((512,512), resample=Image.BILINEAR)
    imageResized.save(outputImagePath)
    return 0

def processFolders(imagesUEFolderPath, masksUEFolderPath ,imagesFolderPath, masksFolderPath, numWorkers=None):

    tasks = buildTasks(imagesUEFolderPath, masksUEFolderPath ,imagesFolderPath, masksFolderPath)

    with ProcessPoolExecutor(max_workers=numWorkers) as executor:
        list(
            tqdm(
                executor.map(dataProcess, tasks),
                total=len(tasks),
                desc="Processing images and masks"
            )
        )

def buildTasks(imagesUEFolderPath, masksUEFolderPath ,imagesFolderPath, masksFolderPath):
    tasks = []

    for filename in sorted(os.listdir(masksUEFolderPath)):
        if filename.lower().endswith(".png"):
            tasks.append((
                os.path.join(masksUEFolderPath, filename),
                os.path.join(masksFolderPath, filename),
                os.path.join(imagesUEFolderPath, filename),
                os.path.join(imagesFolderPath, filename)
            ))
    return tasks


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    imagesUEFolderPath = os.path.join(base_dir, "camera_pose/renderUE/images/")
    masksUEFolderPath = os.path.join(base_dir, "camera_pose/renderUE/masks/")
    imagesFolderPath = os.path.join(base_dir, "camera_pose/datasetProcessedSynthetic/images")
    masksFolderPath = os.path.join(base_dir, "camera_pose/datasetProcessedSynthetic/masks")
    processFolders(imagesUEFolderPath, masksUEFolderPath, imagesFolderPath, masksFolderPath, os.cpu_count())

