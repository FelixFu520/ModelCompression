import os
import numpy as np
import cv2


if __name__ == "__main__":
    dataset_dir = r"D:\Work\ModelCompression\Datasets\wafer\origin\train_no_bg"
    dataset_labels = r"D:\Work\ModelCompression\Datasets\wafer\origin\train_no_bg\label.txt"

    # label info
    labels_info = dict()
    with open(dataset_labels, 'r', encoding='utf8') as f:
        for line in f.readlines():
            name = line.strip().split()[0]
            label = line.strip().split()[1]
            labels_info[label] = name
    print(f"label info of dataset is : {labels_info}")

    # all mask
    all_images_path = [img_p for img_p in os.listdir(dataset_dir) if img_p.endswith("mask.png")]

    # analysis dataset
    histogram_all_images = []
    for img_p in all_images_path:
        image_path = os.path.join(dataset_dir, img_p)
        image = cv2.imread(image_path, -1)
        histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
        histogram_all_images.append(histogram)
    histogram = np.sum(histogram_all_images, axis=0)

    # print
    for key, value in labels_info.items():
        print(f"{value} has {histogram[int(key)]} pixels.")

    """
    train_include_bg分析结果:
    label info of dataset is : {'0': 'background', '1': '凸起', '2': '凹坑', '3': '划伤', '4': '尘点'}
    background has [5.7646534e+08] pixels.
    凸起 has [3465.] pixels.
    凹坑 has [16575.] pixels.
    划伤 has [1661988.] pixels.
    尘点 has [15197.] pixels.
    
    train_no_bg分析结果:
    label info of dataset is : {'0': 'background', '1': '凸起', '2': '凹坑', '3': '划伤', '4': '尘点'}
    background has [5.2279805e+08] pixels.
    凸起 has [3465.] pixels.
    凹坑 has [16575.] pixels.
    划伤 has [1661988.] pixels.
    尘点 has [15197.] pixels.
    """