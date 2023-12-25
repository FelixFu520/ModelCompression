import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil


def is_bg(img_path: str) -> bool:
    image = Image.open(img_path)
    _is_bg = np.max(image) == 0
    return _is_bg


if __name__ == "__main__":
    crop_dir = r"D:\Work\ModelCompression\Datasets\wafer\crop"
    dst_dir = r"D:\Work\ModelCompression\Datasets\wafer\filter"

    all_crop_images = [p for p in os.listdir(crop_dir) if p.endswith("bmp")]

    ng_dir = os.path.join(dst_dir, "ng")
    bg_dir = os.path.join(dst_dir, "bg")
    os.makedirs(ng_dir, exist_ok=True)
    os.makedirs(bg_dir, exist_ok=True)

    for image_p in tqdm(all_crop_images):
        image_path = os.path.join(crop_dir, image_p)
        mask_path = image_path[:-4] + ".png"
        mask_show_path = image_path[:-4] + ".jpg"

        dst_image_path_ng = os.path.join(ng_dir, image_p)
        dst_mask_path_ng = dst_image_path_ng[:-4] + ".png"
        dst_mask_show_path_ng = dst_image_path_ng[:-4] + ".jpg"
        dst_image_path_bg = os.path.join(bg_dir, image_p)
        dst_mask_path_bg = dst_image_path_bg[:-4] + ".png"
        dst_mask_show_path_bg = dst_image_path_bg[:-4] + ".jpg"
        if not is_bg(mask_path):
            shutil.copy(image_path, dst_image_path_ng)
            shutil.copy(mask_path, dst_mask_path_ng)
            shutil.copy(mask_show_path, dst_mask_show_path_ng)
        else:
            shutil.copy(image_path, dst_image_path_bg)
            shutil.copy(mask_path, dst_mask_path_bg)
            shutil.copy(mask_show_path, dst_mask_show_path_bg)

