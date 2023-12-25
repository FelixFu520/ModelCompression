import os
import numpy as np
import glob
import cv2
import shutil
from PIL import Image
from tqdm import tqdm


def slide_crop_image_no_pad(img_path: str, crop_size: list, step_size: list, save_dir: str):
    """
    裁剪图片, 仅裁剪正方形
    :param img_path: 图片路径
    :param crop_size: 裁图高宽
    :param step_size: 步长高宽
    :param save_dir: 存储路径
    :return:
    """
    assert isinstance(crop_size, list) and len(crop_size) == 2
    assert isinstance(step_size, list) and len(step_size) == 2

    mask_path = img_path[:-4] + "_mask.png"
    mask_show_path = img_path[:-4] + "_mask_show.png"

    assert os.path.exists(img_path)
    assert os.path.exists(mask_path)
    assert os.path.exists(mask_show_path)

    image = Image.open(img_path)
    mask = Image.open(mask_path)
    mask_show = Image.open(mask_show_path)

    width, height = image.size

    for y in range(0, height - crop_size[0] + 1, step_size[0]):
        for x in range(0, width - crop_size[1] + 1, step_size[1]):
            box = (x, y, x + crop_size[0], y + crop_size[1])

            image_save_name = os.path.join(save_dir, os.path.basename(img_path)[:-4] + f"_x{x}_y{y}.bmp")
            mask_save_name = os.path.join(save_dir, os.path.basename(img_path)[:-4] + f"_x{x}_y{y}.png")
            mask_show_save_name = os.path.join(save_dir, os.path.basename(img_path)[:-4] + f"_x{x}_y{y}.jpg")

            image.crop(box).save(image_save_name)
            mask.crop(box).save(mask_save_name)
            mask_show.crop(box).save(mask_show_save_name)


if __name__ == '__main__':
    dataset_dir = r"D:\Work\ModelCompression\Datasets\wafer\origin\train_no_bg"
    dst_dir = r"D:\Work\ModelCompression\Datasets\wafer\crop"

    all_images = [p for p in os.listdir(dataset_dir) if p.endswith("bmp")]

    crop_size = [256, 256]
    step_size = [int(256*0.9), int(256*0.9)]
    for img_path in tqdm(all_images):
        image_path = os.path.join(dataset_dir, img_path)
        slide_crop_image_no_pad(image_path, crop_size, step_size, dst_dir)
