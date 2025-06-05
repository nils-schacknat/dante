from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
import torch
import cv2
from tqdm import tqdm
from trt_model import load_backbone


def get_representative_indices(video_path, out_path, backbone, k):
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

            # Run inference to get a single‐channel mask
            frame_transformed = backbone.transform(frame_rgb)
            features, cls_token = backbone.infer(frame_transformed)
            cls_token_list.append(cls_token.cpu().numpy())

            pbar.update(1)

    cls_token_arr = np.concatenate(cls_token_list, axis=0)
    cls_token_arr = cls_token_arr / np.linalg.norm(cls_token_arr, axis=-1, keepdims=True)

    # Run k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(cls_token_arr)

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

    for idx in sorted(idx_list):
        # Seek the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        # Write the frame to disk
        cv2.imwrite(out_path / f"frame_{idx:0{len(str(total_frames))}d}.png", frame)

    cap.release()


if __name__ == "__main__":
    input_size = (518, 924)
    device = 0

    backbone = load_backbone(input_size=input_size, device=device)
    video_path = "assets/stuttgart_university.mp4"
    out_path = Path("selected_images/")
    out_path.mkdir(exist_ok=True)
    get_representative_indices(video_path=video_path, out_path=out_path, backbone=backbone, k=8)
