import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
import json
import random
from tqdm import tqdm
from statistics import mean

import torch
import numpy as np

from sam import sam_model_registry
from sam.utils.transforms import ResizeLongestSide


def get_training_files(path):
    image_dir = os.path.join(path, "images")
    img_all = [os.path.join(image_dir, img) for img in sorted(os.listdir(image_dir))]
    return img_all


def pad_to_square(image, size=1024):
    """Pad an image to make it a square with the given size."""
    h, w = image.shape[:2]

    # Calculate the padding needed
    if h < size:
        pad_h = (size - h) // 2
    else:
        pad_h = 0

    if w < size:
        pad_w = (size - w) // 2
    else:
        pad_w = 0

    # Apply padding if necessary
    padded_image = cv2.copyMakeBorder(
        image,
        pad_h,
        size - h - pad_h if h < size else 0,
        pad_w,
        size - w - pad_w if w < size else 0,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )
    return padded_image


def main():
    sam_checkpoint = "./checkpoints/vit_h.pth"
    model_type = "vit_h"
    device = "cuda"
    training_path = "./datasets/train"
    sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam_model.to(device=device)
    sam_model.train()
    print(
        "Params: {}M".format(
            sum(p.numel() for p in sam_model.mask_decoder.parameters()) / 1e6
        )
    )

    # Set the optimizer and loss
    lr = 1e-4
    optimizer = torch.optim.Adam(
        sam_model.mask_decoder.parameters(), lr=lr, weight_decay=0
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Load bounding boxes
    with open("./datasets/sam_train.json", "r") as f:
        meta = json.load(f)

    # Get the training files
    img_all = get_training_files(training_path)

    # Ensure log directory exists
    log_dir = "./img_logs_sam"
    os.makedirs(log_dir, exist_ok=True)

    # Start training
    num_epochs = 10
    for epoch in range(num_epochs):
        epoch_losses = []
        random.shuffle(img_all)  # Random shuffle the training files
        lab_all = [
            p.replace("images", "labels").replace(".jpg", ".png") for p in img_all
        ]

        img_all_pbar = tqdm(img_all)
        for i, img_path in enumerate(img_all_pbar):
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = cv2.imread(lab_all[i])
            label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

            # Transform - Resize image to have the longest side 1024, then pad to 1024x1024
            sam_trans = ResizeLongestSide(1024)
            resize_image = sam_trans.apply_image(image)
            padded_image = pad_to_square(resize_image)

            resize_label = sam_trans.apply_image(label)
            padded_label = pad_to_square(resize_label)

            image_tensor = torch.as_tensor(padded_image, device=device)
            input_image_torch = image_tensor.permute(2, 0, 1).contiguous()[
                None, :, :, :
            ]

            input_image = sam_model.preprocess(input_image_torch)
            original_image_size = image.shape[:2]
            input_size = tuple(input_image_torch.shape[-2:])

            file_name = os.path.basename(img_path).replace("jpg", "png")

            # Access bounding boxes from the metadata
            bboxes = meta[file_name]["bbox"]
            bboxes = np.array(bboxes)

            with torch.no_grad():
                box = sam_trans.apply_boxes(bboxes, original_image_size)
                box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
                if len(box_torch.shape) == 2:
                    box_torch = box_torch[:, None, :]  # (B, 1, 4)

                image_embedding = sam_model.image_encoder(input_image)
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None,
                    boxes=box_torch,
                    masks=None,
                )

            low_res_masks, iou_predictions = sam_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            low_res_masks = torch.sum(low_res_masks, dim=0, keepdim=True)
            upscaled_masks = sam_model.postprocess_masks(
                low_res_masks, input_size, original_image_size
            ).to(device)

            # Ensure upscaled_masks matches gt_binary_mask size
            gt_mask_resized = torch.from_numpy(
                np.resize(
                    padded_label, (1, 1, padded_label.shape[0], padded_label.shape[1])
                )
            ).to(device)

            upscaled_masks_resized = torch.nn.functional.interpolate(
                upscaled_masks,
                size=gt_mask_resized.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)

            loss = loss_fn(upscaled_masks_resized, gt_binary_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            if i % 100 == 0:
                img_all_pbar.set_postfix(loss=mean(epoch_losses))

                # Save log images for debugging
                image_save = cv2.cvtColor(padded_image, cv2.COLOR_RGB2BGR)
                mask_save = (
                    (upscaled_masks_resized > 0.5)[0].detach().squeeze(0).cpu().numpy()
                )
                mask_save = np.array(mask_save * 255).astype(np.uint8)
                mask_save = np.tile(mask_save[:, :, np.newaxis], 3)
                # Pad or resize the mask_save to 1024x1024
                mask_save = pad_to_square(mask_save)
                # Concatenate and ensure the log image is 1024x1024
                _save = np.concatenate((image_save, mask_save), axis=1)
                _save = pad_to_square(_save, size=1024)
                cv2.imwrite(f"{log_dir}/epoch_{epoch}_batch_{i}.jpg", _save)

        print(f"EPOCH: {epoch}  Mean loss: {mean(epoch_losses)}")
        torch.save(sam_model.state_dict(), f"./checkpoints/{epoch}.pth")


if __name__ == "__main__":
    main()
