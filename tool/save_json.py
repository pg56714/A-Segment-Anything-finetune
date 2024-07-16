import json
import os
import cv2
from skimage import measure
import numpy as np
from tqdm import tqdm


def main():
    path = "./data/train"
    videolists = sorted(os.listdir(os.path.join(path, "labels")))

    # get label path
    meta = {}
    for video in tqdm(videolists):
        meta[video] = {}
        v_path = os.path.join(path, "labels", video)
        lab_all = sorted(os.listdir(v_path))  # 当前video的frame1
        for video_file in lab_all:
            meta[video][video_file] = {}
            lab_path = os.path.join(v_path, video_file)
            label = cv2.imread(lab_path)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

            # get one hot mask
            labels, num = measure.label(label, connectivity=2, return_num=True)
            properties = measure.regionprops(labels)
            valid_label = set()
            for prop in properties:
                if prop.area > 50:
                    valid_label.add(prop.label)
            valid_label = np.array(list(valid_label))
            one_hot_mask = labels[None, :, :] == valid_label[:, None, None]

            if len(valid_label) >= 8 or len(valid_label) == 0:
                # extract bbox
                meta[video][video_file]["bbox"] = []
                y_indices, x_indices = np.where(label > 0)
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                # add perturbation to bounding box coordinates
                H, W = label.shape
                # x_min = max(0, x_min - np.random.randint(0, 20))
                # x_max = min(W, x_max + np.random.randint(0, 20))
                # y_min = max(0, y_min - np.random.randint(0, 20))
                # y_max = min(H, y_max + np.random.randint(0, 20))
                x_min = max(0, x_min)
                x_max = min(W, x_max)
                y_min = max(0, y_min)
                y_max = min(H, y_max)
                bboxes = np.array([x_min, y_min, x_max, y_max])
                meta[video][video_file]["bbox"].append(bboxes.tolist())
                # extract point
                meta[video][video_file]["points"] = []
                meta[video][video_file]["label"] = []
                for i in range(region_num):
                    _mask = one_hot_mask[i]
                    # get point
                    point = np.argwhere(_mask > 0.5)  # (y,x)
                    index = np.random.randint(0, point.shape[0], size=10)
                    point = point[index, :]
                    point = point[:, [1, 0]]  # (x,y)
                    meta[video][video_file]["points"].append(point.tolist())
                    meta[video][video_file]["label"].append(np.array([1] * 10).tolist())
                meta[video][video_file]["shadow_num"] = len(valid_label)

            else:
                # extract bbox
                meta[video][video_file]["bbox"] = []
                region_num = one_hot_mask.shape[0]
                for i in range(region_num):
                    _mask = one_hot_mask[i]
                    # get bbox
                    y_indices, x_indices = np.where(_mask > 0)
                    x_min, x_max = np.min(x_indices), np.max(x_indices)
                    y_min, y_max = np.min(y_indices), np.max(y_indices)
                    # add perturbation to bounding box coordinates
                    H, W = _mask.shape
                    x_min = max(0, x_min - np.random.randint(0, 20))
                    x_max = min(W, x_max + np.random.randint(0, 20))
                    y_min = max(0, y_min - np.random.randint(0, 20))
                    y_max = min(H, y_max + np.random.randint(0, 20))
                    bboxes = np.array([x_min, y_min, x_max, y_max])
                    meta[video][video_file]["bbox"].append(bboxes.tolist())
                # extract point
                meta[video][video_file]["points"] = []
                meta[video][video_file]["label"] = []
                for i in range(region_num):
                    _mask = one_hot_mask[i]
                    # get point
                    point = np.argwhere(_mask > 0.5)  # (y,x)
                    index = np.random.randint(0, point.shape[0], size=10)
                    point = point[index, :]
                    point = point[:, [1, 0]]  # (x,y)
                    meta[video][video_file]["points"].append(point.tolist())
                    meta[video][video_file]["label"].append(np.array([1] * 10).tolist())
                meta[video][video_file]["shadow_num"] = len(valid_label)

    b = json.dumps(meta, indent=4)
    f2 = open("sam_train.json", "w")
    f2.write(b)
    f2.close()
    # print()


if __name__ == "__main__":
    main()
