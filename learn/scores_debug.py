import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score


def evaluate_masks(true_mask, pred_mask):
    # Normalize masks to binary values (0 and 1)
    true_flat = (true_mask.flatten() / 255).astype(np.uint8)
    pred_flat = (pred_mask.flatten() / 255).astype(np.uint8)
    # print("True mask:", true_mask)
    # print(true_flat)
    # print(pred_mask.flatten())

    # Calculate precision, recall, and F1 scores
    precision = precision_score(true_flat, pred_flat)
    recall = recall_score(true_flat, pred_flat)
    f1 = f1_score(true_flat, pred_flat)
    jaccard = jaccard_score(true_flat, pred_flat)

    print(f"精確度 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1 分數 (F1 Score): {f1:.4f}")
    print(f"Jaccard 指數 (Jaccard Index = iou): {jaccard:.4f}")


# equal to f1 score
def calculate_dsc(mask1, mask2):
    """Calculate the Dice Similarity Coefficient (DSC) between two masks."""
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)

    intersection = np.logical_and(mask1, mask2).sum()
    total = mask1.sum() + mask2.sum()
    dsc = (2.0 * intersection) / total if total != 0 else 0
    return dsc


# def convert_color_mask_to_binary(color_mask):
#     if color_mask is None:
#         raise ValueError("Image not loaded. Check file path and integrity.")
#     gray_mask = cv2.cvtColor(color_mask, cv2.COLOR_BGR2GRAY)
#     binary_mask = cv2.adaptiveThreshold(
#         gray_mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
#     )
#     return binary_mask


def convert_to_white_mask(color_mask):
    gray_mask = cv2.cvtColor(color_mask, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)
    return white_mask


# def convert_to_white_mask_to_binary(color_mask):
#     gray_mask = cv2.cvtColor(color_mask, cv2.COLOR_BGR2GRAY)
#     _, white_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)

#     binary_mask = cv2.adaptiveThreshold(
#         white_mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
#     )
#     return binary_mask


def visualize_masks(image1, image2, title1="Mask 1", title2="Mask 2"):
    """Visualize two images side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image1, cmap="gray")
    axes[0].set_title(title1)
    axes[1].imshow(image2, cmap="gray")
    axes[1].set_title(title2)
    plt.show()


def check_alignment_and_size(mask1, mask2):
    print("Mask1 shape:", mask1.shape)
    print("Mask2 shape:", mask2.shape)
    if mask1.shape != mask2.shape:
        print("Warning: Masks have different dimensions.")
    else:
        print("Masks are of the same dimension.")


def visualize_differences(mask1, mask2):
    difference = cv2.absdiff(mask1, mask2)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(mask1, cmap="gray")
    plt.title("Mask 1")
    plt.subplot(1, 3, 2)
    plt.imshow(mask2, cmap="gray")
    plt.title("Mask 2")
    plt.subplot(1, 3, 3)
    plt.imshow(difference, cmap="hot")
    plt.title("Differences")
    plt.show()


def clean_mask(binary_mask):
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    return cleaned_mask


def calculate_ciou(mask1, mask2):
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0

    # Check if either mask has no non-zero elements
    mask1_nonzero = mask1.nonzero()
    mask2_nonzero = mask2.nonzero()

    if mask1_nonzero[0].size == 0 or mask2_nonzero[0].size == 0:
        return 0.0

    mask1_center = np.mean(np.argwhere(mask1 == 1), axis=0)
    mask2_center = np.mean(np.argwhere(mask2 == 1), axis=0)
    distance = np.linalg.norm(mask1_center - mask2_center)
    enclose_x_min = min(mask1.nonzero()[0].min(), mask2.nonzero()[0].min())
    enclose_x_max = max(mask1.nonzero()[0].max(), mask2.nonzero()[0].max())
    enclose_y_min = min(mask1.nonzero()[1].min(), mask2.nonzero()[1].min())
    enclose_y_max = max(mask1.nonzero()[1].max(), mask2.nonzero()[1].max())
    c_diag = np.linalg.norm(
        np.array([enclose_x_max, enclose_y_max])
        - np.array([enclose_x_min, enclose_y_min])
    )
    mask1_shape = mask1.shape[0] / mask1.shape[1]
    mask2_shape = mask2.shape[0] / mask2.shape[1]
    v = (4 / np.pi**2) * (np.arctan(mask1_shape) - np.arctan(mask2_shape)) ** 2

    denominator = 1 - intersection / union + v
    if denominator != 0:
        alpha = v / denominator
    else:
        alpha = 0

    ciou = (intersection / union) - (distance**2 / c_diag**2 + alpha * v)
    return ciou


def main():
    # Paths to the predicted and ground truth mask images
    pred_mask_path = "./results/sam/labels/images/lssd148.png"
    true_mask_path = "./datasets/test/labels/lssd148.png"

    # Check if files exist
    if not os.path.exists(pred_mask_path):
        print(f"Predicted mask file does not exist: {pred_mask_path}")
        return

    if not os.path.exists(true_mask_path):
        print(f"True mask file does not exist: {true_mask_path}")
        return

    # Load and convert the masks to binary
    pred_mask = convert_to_white_mask(cv2.imread(pred_mask_path))
    true_mask = convert_to_white_mask(cv2.imread(true_mask_path))

    # Clean masks
    pred_mask = clean_mask(pred_mask)
    true_mask = clean_mask(true_mask)

    # Diagnostics to check mask alignment and size
    check_alignment_and_size(pred_mask, true_mask)

    # Calculate DSC
    dsc = calculate_dsc(pred_mask, true_mask)
    print(f"Dice Similarity Coefficient (DSC): {dsc:.4f}")

    # Visualize the masks and differences
    visualize_masks(pred_mask, true_mask, "Predicted Mask", "Ground Truth Mask")
    visualize_differences(pred_mask, true_mask)

    evaluate_masks(true_mask, pred_mask)

    ciou = calculate_ciou(pred_mask, true_mask)
    print(f"Complete Intersection over Union (ciou): {ciou:.4f}")


if __name__ == "__main__":
    main()
