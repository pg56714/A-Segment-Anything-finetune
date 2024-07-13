import os
import torch
import numpy as np
from skimage import measure
import gradio as gr
from sam import sam_model_registry
from sam.utils.transforms import ResizeLongestSide

# Initialize SAM model
sam_checkpoint = "./checkpoints/Incremental-Learning00.pth"
model_type = "vit_b"
sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)

device = "cuda"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sam_model.to(device=device)
sam_model.eval()


def generate_mask_sam(first_frame):

    # ImageEditor
    # if "composite" not in first_frame:
    #     raise ValueError("Composite image data not found in the input.")

    # first_frame_image = np.array(first_frame["composite"])[
    #     :, :, :3
    # ]  # Extract RGB channels

    first_frame_image = np.array(first_frame)[:, :, :3]

    if first_frame_image.ndim != 3 or first_frame_image.shape[2] != 3:
        raise ValueError("Input image must be an RGB image.")

    # print("Shape of first_frame_image:", first_frame_image.shape)
    # print("Data type of first_frame_image:", first_frame_image.dtype)

    # Set bounding box to cover the entire image
    H, W, _ = first_frame_image.shape
    bboxes = np.array([[0, 0, W, H]])  # Full image bounding box

    sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
    resize_image = sam_trans.apply_image(first_frame_image)
    image_tensor = torch.as_tensor(resize_image, device=device)
    input_image_torch = image_tensor.permute(2, 0, 1).contiguous().unsqueeze(0)
    input_image = sam_model.preprocess(input_image_torch)
    original_image_size = first_frame_image.shape[:2]
    input_size = input_image_torch.shape[-2:]

    with torch.no_grad():
        box = sam_trans.apply_boxes(bboxes, original_image_size)
        box_torch = torch.as_tensor(box, dtype=torch.float, device=device).unsqueeze(
            0
        )  # Ensure it has the correct shape

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

    # multimask_output=True
    # print("Shape of low_res_masks:", low_res_masks.shape)
    # print("Mask values before thresholding:", low_res_masks)
    # processed_masks = torch.sigmoid(low_res_masks)
    # combined_mask = torch.max(processed_masks[0], dim=0).values
    # threshold = 0.5
    # binary_mask = (combined_mask > threshold).type(torch.uint8) * 255

    # upscaled_mask = (
    #     torch.nn.functional.interpolate(
    #         binary_mask.unsqueeze(0).unsqueeze(0).float(),
    #         size=original_image_size,
    #         mode="nearest",
    #     )
    #     .byte()
    #     .squeeze()
    # )

    # thresholded_mask = upscaled_mask.cpu().numpy()
    # if thresholded_mask.ndim == 3 and thresholded_mask.shape[0] == 1:
    #     thresholded_mask = thresholded_mask.squeeze(0)

    # multimask_output = False
    low_res_masks = torch.sum(low_res_masks, dim=0, keepdim=True)
    upscaled_masks = sam_model.postprocess_masks(
        low_res_masks, input_size, original_image_size
    ).to(device)
    thresholded_mask = (
        (upscaled_masks > 0.4)[0].detach().squeeze(0).cpu().numpy()
    )  # Adjusted threshold
    thresholded_mask = np.array(thresholded_mask * 255).astype(
        np.uint8
    )  # Convert mask to uint8 format for display

    return thresholded_mask  # Directly return the model-generated mask


title = """<p><h1 align="center">AnyShadow</h1></p>"""
description = """<p>Gradio demo for Shadow Detection<p>"""

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            first_frame = gr.Image()
            # first_frame = gr.ImageEditor(interactive=True)
            first_mask_button = gr.Button("Submit")
        first_mask = gr.Image()
    first_mask_button.click(
        fn=generate_mask_sam, inputs=first_frame, outputs=first_mask
    )

demo.launch(debug=True, show_error=True)
