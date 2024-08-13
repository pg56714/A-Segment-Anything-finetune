import os
import json
from tqdm import tqdm
import cv2


def draw_annotations(image_path, annotations, save_dir):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image loading failed, check the path：{image_path}")
        return

    for bbox in annotations["bbox"]:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    for points in annotations["points"]:
        for x, y in points:
            cv2.circle(image, (x, y), 3, (255, 0, 0), -1)

    save_path = os.path.join(save_dir, os.path.basename(image_path))
    cv2.imwrite(save_path, image)


def main():
    with open("./sam_train.json", "r") as file:
    # with open("./sam_test.json", "r") as file:
    # with open("./sam_challenge.json", "r") as file:
        data = json.load(file)
        print(f"Total keys in JSON: {len(data.keys())}")

    base_save_dir = "./labels"
    if not os.path.exists(base_save_dir):
        os.makedirs(base_save_dir)

    for image_file, annotations in tqdm(data.items(), desc="Processing images"):
        image_path = os.path.abspath(f"./data_only_full_mask/train/labels/{image_file}")
        # image_path = os.path.abspath(f"./data_only_full_mask/test/labels/{image_file}")
        # image_path = os.path.abspath(f"./data_only_full_mask/challenge/labels/{image_file}")
        if os.path.exists(image_path):
            draw_annotations(image_path, annotations, base_save_dir)
        else:
            print(f"File does not exist: {image_path}")


if __name__ == "__main__":
    main()
