from dante import DANTE
from trt_model import load_backbone
import torch
import cv2
from tqdm import tqdm
import numpy as np


def run_inference_on_video(video_path: str, output_path: str, model: DANTE) -> None:
    """
    Runs inference on a video file and writes the results to a new video.

    Args:
        video_path: Path to the video file.
        output_path: Path to the output video.
        model: DANTE model.
    """
    # Open the source video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Grab original properties
    fps      = cap.get(cv2.CAP_PROP_FPS)
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter using the same FPS and frame size
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=total_frames, desc="Running inference") as pbar:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Run inference to get a single‐channel mask
            mask = (model.infer_image(frame_rgb) * 255).astype(np.uint8)

            # Colorize the mask and overlay onto the original BGR frame
            mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            alpha = 0.4
            overlayed_bgr = cv2.addWeighted(mask_colored, alpha, frame_bgr, 1 - alpha, 0)

            # Write the overlayed BGR frame into the new video
            out.write(overlayed_bgr)

            pbar.update(1)

    # Release resources
    cap.release()
    out.release()



if __name__ == "__main__":
    # The input dimensions the backbone has been compiled with
    # during inference we maintain the aspect ratio
    input_size = (518, 924)
    # The gpu the backbone and model should run on (if multiple are available)
    device = 0

    backbone = load_backbone(input_size=input_size, device=device)
    dante = DANTE(backbone=backbone)

    # Load the pretrained checkpoint (run train.py)
    checkpoint_path = "trained_decoder.pth"
    dante.linear.load_state_dict(torch.load(checkpoint_path))
    dante = dante.to(backbone.device)

    video_path = "assets/stuttgart_university_campus.mp4"
    output_path = "video_processed.mp4"
    run_inference_on_video(video_path, output_path, dante)

    