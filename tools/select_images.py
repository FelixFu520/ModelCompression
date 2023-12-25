import os
import shutil
import random


if __name__ == "__main__":
    bg_dir = r"D:\Work\ModelCompression\Datasets\wafer\filter\bg"
    ng_dir = r"D:\Work\ModelCompression\Datasets\wafer\filter\ng"

    train_dir = r"D:\Work\ModelCompression\Datasets\wafer\data\train"
    val_dir = r"D:\Work\ModelCompression\Datasets\wafer\data\val"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    bg_list = [os.path.join(bg_dir, p) for p in os.listdir(bg_dir) if p.endswith("bmp")]
    ng_list = [os.path.join(ng_dir, p) for p in os.listdir(ng_dir) if p.endswith("bmp")]
    random.shuffle(bg_list)
    random.shuffle(ng_list)

    print(f"BG images:{len(bg_list)}")
    print(f"NG images:{len(ng_list)}")

    train_ng_num = int(len(ng_list) * 0.8)
    train_bg_num = int(train_ng_num * 0.2)
    train_num = train_bg_num + train_ng_num

    val_ng_num = len(ng_list) - train_ng_num
    val_bg_num = int(val_ng_num * 0.2)
    val_num = val_bg_num + val_ng_num

    print(f"训练集数量:{train_num}, NG:{train_ng_num}, BG:{train_bg_num}")
    print(f"验证集数量:{val_num}, NG:{val_ng_num}, BG:{val_bg_num}")

    for i in range(len(ng_list)):
        if i < train_ng_num:
            shutil.copy(ng_list[i], os.path.join(train_dir, os.path.basename(ng_list[i])))
            shutil.copy(ng_list[i][:-4]+".png", os.path.join(train_dir, os.path.basename(ng_list[i]))[:-4]+".png")
            shutil.copy(ng_list[i][:-4]+".jpg", os.path.join(train_dir, os.path.basename(ng_list[i]))[:-4]+".jpg")
        else:
            shutil.copy(ng_list[i], os.path.join(val_dir, os.path.basename(ng_list[i])))
            shutil.copy(ng_list[i][:-4] + ".png", os.path.join(val_dir, os.path.basename(ng_list[i]))[:-4] + ".png")
            shutil.copy(ng_list[i][:-4] + ".jpg", os.path.join(val_dir, os.path.basename(ng_list[i]))[:-4] + ".jpg")

    for i in range(len(bg_list)):
        if i < train_bg_num:
            shutil.copy(bg_list[i], os.path.join(train_dir, os.path.basename(bg_list[i])))
            shutil.copy(bg_list[i][:-4]+".png", os.path.join(train_dir, os.path.basename(bg_list[i]))[:-4]+".png")
            shutil.copy(bg_list[i][:-4]+".jpg", os.path.join(train_dir, os.path.basename(bg_list[i]))[:-4]+".jpg")
        elif i < train_bg_num+val_bg_num:
            shutil.copy(bg_list[i], os.path.join(val_dir, os.path.basename(bg_list[i])))
            shutil.copy(bg_list[i][:-4] + ".png", os.path.join(val_dir, os.path.basename(bg_list[i]))[:-4] + ".png")
            shutil.copy(bg_list[i][:-4] + ".jpg", os.path.join(val_dir, os.path.basename(bg_list[i]))[:-4] + ".jpg")
        else:
            break
