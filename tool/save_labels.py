import os
import json
from tqdm import tqdm
import cv2


def draw_annotations(image_path, annotations, save_dir):
    # 讀取圖片
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image loading failed, check the path：{image_path}")
        return

    # 繪製邊界框
    for bbox in annotations["bbox"]:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # 繪製特徵點
    for points in annotations["points"]:
        for x, y in points:
            cv2.circle(image, (x, y), 3, (255, 0, 0), -1)

    # 儲存圖片
    save_path = os.path.join(save_dir, os.path.basename(image_path))
    cv2.imwrite(save_path, image)


def main():
    print("Current Working Directory: ", os.getcwd())

    # 載入 JSON 檔案
    with open("../datasets/sam_train.json", "r") as file:
        data = json.load(file)
        print(f"Total keys in JSON: {len(data.keys())}")

    base_save_dir = "./labels"  # 指定保存圖片的根資料夾

    # 處理每個檔案的標註
    for main_key in tqdm(data.keys(), desc="Processing keys"):  # 使用主鍵動態讀取
        # 為每個主鍵創建一個子資料夾
        save_dir = os.path.join(base_save_dir, main_key)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 使用 tqdm 顯示進度條
        for image_file, annotations in data[main_key].items():
            image_path = os.path.abspath(
                f"../datasets/train/labels/{main_key}/{image_file}"
            )
            if os.path.exists(image_path):
                draw_annotations(image_path, annotations, save_dir)
            else:
                print(f"File does not exist：{image_path}")


if __name__ == "__main__":
    main()
