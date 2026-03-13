import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from utils.preprocessing import dataProcess

def build_tasks(images_ue_folder, masks_ue_folder, images_out_folder, masks_out_folder):
    """
    Construye una lista de tareas con los paths de entrada y salida para imágenes y máscaras.
    """
    tasks = []
    for filename in sorted(os.listdir(masks_ue_folder)):
        if filename.lower().endswith(".png"):
            tasks.append((
                os.path.join(masks_ue_folder, filename),
                os.path.join(masks_out_folder, filename),
                os.path.join(images_ue_folder, filename),
                os.path.join(images_out_folder, filename)
            ))
    return tasks

def process_folders(images_ue_folder, masks_ue_folder, images_out_folder, masks_out_folder, num_workers=None):
    """
    Procesa todas las imágenes y máscaras usando múltiples procesos.
    """

    if not os.path.exists(images_out_folder):
        os.makedirs(images_out_folder)
    if not os.path.exists(masks_out_folder):
        os.makedirs(masks_out_folder)


    tasks = build_tasks(images_ue_folder, masks_ue_folder, images_out_folder, masks_out_folder)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(
            tqdm(
                executor.map(dataProcess, *zip(*tasks)),
                total=len(tasks),
                desc="Processing images and masks"
            )
        )

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_ue_folder = os.path.join(script_dir, "..", "data", "renderUE", "images")
    masks_ue_folder = os.path.join(script_dir, "..", "data", "renderUE", "masks")
    images_out_folder = os.path.join(script_dir, "..", "data", "datasetPreProcessed", "images")
    masks_out_folder = os.path.join(script_dir, "..", "data", "datasetPreProcessed", "masks")
    process_folders(images_ue_folder, masks_ue_folder, images_out_folder, masks_out_folder, os.cpu_count())

