import os
import cv2
from tqdm import tqdm

def convert_to_white_mask(image_path, save_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image {image_path}")
        return
    gray_mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)
    cv2.imwrite(save_path, white_mask)

def batch_convert_to_white_masks(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    files = [f for f in os.listdir(input_directory) if f.endswith('.png')]
    for filename in tqdm(files, desc="Converting images"):
        file_path = os.path.join(input_directory, filename)
        save_path = os.path.join(output_directory, filename)
        convert_to_white_mask(file_path, save_path)

# print("Current working directory:", os.getcwd())
input_directory = os.path.abspath('../datasets/train/labels')
output_directory = os.path.abspath('../labels')
batch_convert_to_white_masks(input_directory, output_directory)
