import cv2
import numpy as np
import matplotlib.pyplot as plt


def convert_color_mask_to_binary(color_mask):
    if color_mask is None:
        raise ValueError("Image not loaded. Check file path and integrity.")

    # Convert to grayscale and apply adaptive threshold
    gray_mask = cv2.cvtColor(color_mask, cv2.COLOR_BGR2GRAY)
    binary_mask = cv2.adaptiveThreshold(
        gray_mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return binary_mask


def convert_to_white_mask(color_mask):
    gray_mask = cv2.cvtColor(color_mask, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)
    return white_mask


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


def visualize_differences(mask1, mask2, title1="Mask 1", title2="Mask 2"):
    difference = cv2.absdiff(mask1, mask2)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(mask1, cmap="gray")
    plt.title(title1)
    plt.subplot(1, 3, 2)
    plt.imshow(mask2, cmap="gray")
    plt.title(title2)
    plt.subplot(1, 3, 3)
    plt.imshow(difference, cmap="hot")
    plt.title("Differences")
    plt.show()


def clean_mask(binary_mask):
    kernel = np.ones((5, 5), np.uint8)

    # 進行膨脹操作
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)

    # 進行腐蝕操作
    cleaned_mask = cv2.erode(dilated_mask, kernel, iterations=1)

    cleaned_mask1 = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # 繪製過程
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(binary_mask, cmap="gray")
    plt.title("Original Mask")

    plt.subplot(1, 4, 2)
    plt.imshow(dilated_mask, cmap="gray")
    plt.title("Dilated Mask")

    plt.subplot(1, 4, 3)
    plt.imshow(cleaned_mask, cmap="gray")
    plt.title("Cleaned Mask (Closed)")

    plt.subplot(1, 4, 4)
    plt.imshow(cleaned_mask1, cmap="gray")
    plt.title("Cleaned Mask1 (Closed)")
    plt.show()

    return cleaned_mask


def main():
    # Paths to the predicted and ground truth mask images
    pred_mask_path = "./results/sam/labels/images/lssd148.png"
    true_mask_path = "./datasets/test/labels/lssd148.png"

    # Load and convert the masks to binary
    # pred_mask = convert_color_mask_to_binary(cv2.imread(pred_mask_path))
    # true_mask = convert_color_mask_to_binary(cv2.imread(true_mask_path))

    pred_mask = convert_to_white_mask(cv2.imread(pred_mask_path))
    true_mask = convert_to_white_mask(cv2.imread(true_mask_path))

    # Clean masks
    pred_mask = clean_mask(pred_mask)
    true_mask = clean_mask(true_mask)

    # Diagnostics to check mask alignment and size
    check_alignment_and_size(pred_mask, true_mask)

    # Visualize the masks and differences
    visualize_masks(pred_mask, true_mask, "Predicted Mask", "Ground Truth Mask")
    visualize_differences(pred_mask, true_mask, "Predicted Mask", "Ground Truth Mask")


if __name__ == "__main__":
    main()
