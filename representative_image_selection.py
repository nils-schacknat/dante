from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
import torch
import cv2
from tqdm import tqdm
from trt_model import load_backbone, TensorRTModel


def get_representative_images(video_path: str, out_path: Path, backbone: TensorRTModel, k: int):
    """
    Obtains representative images from a video sequence and writes them to an output folder.
    The selection process is based on the class token of the provided ViT backbone.

    Args:
        video_path (str): Path to the video sequence.
        out_path (Path): Path to the output folder.
        backbone (TensorRTModel): The compiled ViT backbone.
        k (int): The number of images to select.
    """
    # Open the source video
    cap = cv2.VideoCapture(video_path)

    cls_token_list = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=total_frames, desc="Running inference") as pbar:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Run inference to obtain the class tokens
            frame_transformed = backbone.transform(frame_rgb)
            features, cls_token = backbone.infer(frame_transformed)
            cls_token_list.append(cls_token.cpu().numpy())

            pbar.update(1)

    # Transform the class tokens to an array and normalize them
    cls_token_arr = np.concatenate(cls_token_list, axis=0)
    cls_token_arr = cls_token_arr / np.linalg.norm(cls_token_arr, axis=-1, keepdims=True)

    # Run k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(cls_token_arr)

    # Identify the centers
    centers = kmeans.cluster_centers_
    centers = centers / np.linalg.norm(centers, axis=-1, keepdims=True)
    labels = kmeans.labels_

    # For each cluster, find the closest image to the cluster center
    idx_list = []
    for i in range(k):
        cluster_indices = np.where(labels == i)[0]
        cluster_vectors = cls_token_arr[cluster_indices]
        similarities = cluster_vectors @ centers[i]
        closest_idx = cluster_indices[np.argmax(similarities)]
        idx_list.append(closest_idx)

    # Write the selected images to the output directory
    for idx in sorted(idx_list):
        # Seek the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        cv2.imwrite(out_path / f"frame_{idx:0{len(str(total_frames))}d}.png", frame)

    cap.release()


if __name__ == "__main__":
    # The input dimensions the model backbone has been compiled with
    input_size = (518, 924)
    # The gpu the backbone and model should run on (if multiple are available)
    device = 0

    backbone = load_backbone(input_size=input_size, device=device)
    video_path = "assets/stuttgart_university.mp4"
    out_path = Path("selected_images/")
    out_path.mkdir(exist_ok=True)
    get_representative_images(video_path=video_path, out_path=out_path, backbone=backbone, k=8)
