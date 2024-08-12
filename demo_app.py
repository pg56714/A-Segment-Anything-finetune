import os
import torch
import numpy as np
from skimage import measure
import gradio as gr
from sam import sam_model_registry
from sam.utils.transforms import ResizeLongestSide

# Initialize SAM model
sam_checkpoint = "./checkpoints/old/200_0202.pth"
model_type = "vit_b"
sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sam_model.to(device=device)
sam_model.eval()


def generate_mask_sam(frame):
    frame_image = np.array(frame)[:, :, :3]

    H, W, _ = frame_image.shape
    bboxes = np.array([[0, 0, W, H]])

    sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
    resize_image = sam_trans.apply_image(frame_image)
    image_tensor = torch.as_tensor(resize_image, device=device)
    input_image_torch = image_tensor.permute(2, 0, 1).contiguous().unsqueeze(0)
    input_image = sam_model.preprocess(input_image_torch)
    original_image_size = frame_image.shape[:2]
    input_size = input_image_torch.shape[-2:]

    with torch.no_grad():
        box = sam_trans.apply_boxes(bboxes, original_image_size)
        box_torch = torch.as_tensor(box, dtype=torch.float, device=device).unsqueeze(0)

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
    thresholded_mask = (upscaled_masks > 0.5)[0].detach().squeeze(0).cpu().numpy()
    thresholded_mask = np.array(thresholded_mask * 255).astype(np.uint8)

    return thresholded_mask


title = """<p><h1 align="center">Test Demo</h1></p>"""
description = """<p>Test<p>"""

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            frame = gr.Image()
            mask_button = gr.Button("Submit")
        mask = gr.Image()
    mask_button.click(fn=generate_mask_sam, inputs=frame, outputs=mask)

demo.launch(debug=True, show_error=True)
