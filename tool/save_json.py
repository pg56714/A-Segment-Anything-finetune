import os
import cv2
import json
from skimage import measure
import numpy as np
from tqdm import tqdm


def main():
    # base_path = "../datasets/train/labels"
    base_path = "../datasets/test/labels"
    files = sorted(os.listdir(base_path))
    meta = {}

    for file in tqdm(files):
        image_file_path = os.path.join(base_path, file)
        label = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)

        # label = cv2.resize(label, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        labels, num = measure.label(label, connectivity=2, return_num=True)
        properties = measure.regionprops(labels)

        image_data = {"bbox": [], "points": [], "label": []}
        for prop in properties:
            if prop.area > 50:
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
                    )
                    image_data["label"].append([1] * 10)

        meta[file] = image_data

    # with open("../datasets/sam_train.json", "w") as f:
    with open("../datasets/sam_test.json", "w") as f:
        json.dump(meta, f, indent=4)


if __name__ == "__main__":
    main()
