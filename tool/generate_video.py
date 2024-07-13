import os
import torch
import torchvision
import numpy as np
import cv2

# generate video after vsd inference
def generate_video_from_frames(frames, output_path, fps=30):
    """
    Generates a video from a list of frames.
    
    Args:
        frames (list of numpy arrays): The frames to include in the video.
        output_path (str): The path to save the generated video.
        fps (int, optional): The frame rate of the output video. Defaults to 30.
    """

    frames = torch.from_numpy(np.asarray(frames))  # [frame, h, w, c]
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    torchvision.io.write_video(output_path, frames, fps=fps, video_codec="libx264")
    return output_path


video_path = r"/data/wangyh/data4/video_shadow_detection/segment8/41/test/images/bike1"
all_frame = []
for i in sorted(os.listdir(video_path)):
    _frame = cv2.imread(os.path.join(video_path, i))
    _frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB)
    all_frame.append(_frame)

generate_video_from_frames(all_frame, "./bike1.mp4")