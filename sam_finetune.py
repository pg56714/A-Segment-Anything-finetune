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
    videolists = sorted(os.listdir(os.path.join(path, "images")))
    img_all = []
    for video in videolists:
        v_path = os.path.join(path, "images", video)
        imglist = sorted(os.listdir(v_path))  # all frames of current video
        img_all = img_all + [os.path.join(v_path, img) for img in imglist]
    return img_all  # a list of all image paths


def main():
    sam_checkpoint = "./checkpoints/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
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

    # set the optimizer and loss
    lr = 1e-4
    optimizer = torch.optim.Adam(
        sam_model.mask_decoder.parameters(), lr=lr, weight_decay=0
    )
    # loss_fn = torch.nn.MSELoss()
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # load bounding boxes
    with open("./datasets/sam_train.json", "r") as f:
        meta = json.load(f)

    # get the training files
    img_all = get_training_files(training_path)

    # start training!!!
    num_epochs = 10
    for epoch in range(num_epochs):
        epoch_losses = []
        random.shuffle(img_all)  # random shuffle the training files
        lab_all = [
            p.replace("images", "labels").replace(".jpg", ".png") for p in img_all
        ]

        img_all_pbar = tqdm(img_all)
        for i, img_path in enumerate(img_all_pbar):
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = cv2.imread(lab_all[i])
            label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

            # transform
            sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)  # 1024
            resize_image = sam_trans.apply_image(image)  # padding to 1024 * 1024
            image_tensor = torch.as_tensor(resize_image, device=device)
            input_image_torch = image_tensor.permute(2, 0, 1).contiguous()[
                None, :, :, :
            ]

            input_image = sam_model.preprocess(input_image_torch)
            original_image_size = image.shape[:2]
            input_size = tuple(input_image_torch.shape[-2:])

            # video_name = img_path.split("/")[-2]
            # file_name = img_path.split("/")[-1].replace("jpg", "png")

            video_name = os.path.basename(os.path.dirname(img_path))
            file_name = os.path.basename(img_path).replace("jpg", "png")

            # # Debugging print statements
            # print(f"Processing video: {video_name}, file: {file_name}")

            # if video_name not in meta:
            #     print(f"Video {video_name} not found in metadata")
            #     continue

            # if file_name not in meta[video_name]:
            #     print(f"File {file_name} not found in metadata for video {video_name}")
            #     continue

            bboxes = meta[video_name][file_name]["bbox"]
            bboxes = np.array(bboxes)

            with torch.no_grad():
                box = sam_trans.apply_boxes(bboxes, (original_image_size))
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
            gt_mask_resized = torch.from_numpy(
                np.resize(label, (1, 1, label.shape[0], label.shape[1]))
            ).to(device)
            gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)

            loss = loss_fn(upscaled_masks, gt_binary_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            if i % 100 == 0:
                img_all_pbar.set_postfix(loss=mean(epoch_losses))
                image_save = cv2.imread(img_path)
                image_save = cv2.cvtColor(image_save, cv2.COLOR_BGR2RGB)
                mask_save = (upscaled_masks > 0.5)[0].detach().squeeze(0).cpu().numpy()
                mask_save = np.array(mask_save * 255).astype(np.uint8)
                mask_save = np.tile(mask_save[:, :, np.newaxis], 3)
                _save = np.concatenate((image_save, mask_save), axis=1)
                cv2.imwrite("./img_logs_sam/{}_{}.jpg".format(epoch, i), _save)

        print(f"EPOCH: {epoch}  Mean loss: {mean(epoch_losses)}")
        torch.save(sam_model.state_dict(), f"./checkpoints/{epoch}.pth")


if __name__ == "__main__":
    main()
