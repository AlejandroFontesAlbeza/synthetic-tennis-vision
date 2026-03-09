from PIL import Image
import numpy as np
import os

from src.utils.palette import exactColorPalette, rangeColorPalette



def dataProcess(inputMaskPath, outputMaskPath, inputImagePath, outputImagePath):
    """
    Procesa una máscara y una imagen:
    - Convierte la máscara a escala de grises según la paleta y la reescala a 512x512.
    - Reescala la imagen a 512x512.
    """
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