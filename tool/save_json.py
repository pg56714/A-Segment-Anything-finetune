import os
import cv2
import json
from skimage import measure
import numpy as np
from tqdm import tqdm


def main():
    base_path = os.path.abspath("../train/labels")  # 使用絕對路徑以避免路徑錯誤
    folders = sorted(os.listdir(base_path))  # 假設每個 folder 對應一組影像

    meta = {}
    for folder in tqdm(folders):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):  # 確保是資料夾
            files = sorted(os.listdir(folder_path))
            meta[folder] = {}
            for file in files:
                file_path = os.path.join(folder_path, file)
                label = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # 直接讀取灰階圖
                if label is not None:
                    # 使用 skimage.measure.label 分析連通區域
                    labels, num = measure.label(label, connectivity=2, return_num=True)
                    properties = measure.regionprops(labels)
                    image_data = {"bbox": [], "points": [], "label": []}
                    for prop in properties:
                        if prop.area > 50:  # 只處理面積大於50的區域
                            minr, minc, maxr, maxc = prop.bbox
                            image_data["bbox"].append([minc, minr, maxc, maxr])
                            mask = labels == prop.label
                            points = np.argwhere(mask)
                            if points.size > 10:
                                selected_points = points[
                                    np.random.choice(points.shape[0], 10, replace=False)
                                ]
                                image_data["points"].append(
                                    selected_points[:, [1, 0]].tolist()
                                )  # 轉換 x, y 格式
                                image_data["label"].append([1] * 10)
                    meta[folder][file] = image_data
                else:
                    print(f"Error reading file: {file_path}")

    # 寫入 JSON 文件
    with open("../datasets/sam_train.json", "w") as f:
        json.dump(meta, f, indent=4)


if __name__ == "__main__":
    main()
