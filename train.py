from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
import cv2
import torch
import random

from dante import DANTE
from trt_model import load_backbone
import lightning as L
from torch.utils.data import DataLoader, ConcatDataset


def random_square_crop(frame: np.ndarray, mask:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly crop a frame and its corresponding mask to a square of the largest possible size.

    Args:
        frame (np.array): The frame to be cropped.
        mask (np.array): The mask to be cropped.

    Returns:
        tuple[np.ndarray, np.ndarray]: The cropped frame and its corresponding mask.
    """
    # Get dimensions
    h, w = frame.shape[:2]
    crop_size = min(h, w)

    # Randomly select top-left corner for the crop
    x_max = w - crop_size
    y_max = h - crop_size
    x = random.randint(0, x_max)
    y = random.randint(0, y_max)

    # Crop both frame and mask
    frame_cropped = frame[y:y+crop_size, x:x+crop_size]
    mask_cropped = mask[y:y+crop_size, x:x+crop_size]

    return frame_cropped, mask_cropped


class TraversabilityDataset(Dataset):
    """
    Construct a Dataset from annotated frames.
    """
    def __init__(self, data_dir: Path, transform: callable, random_crop_to_square=True):
        """
        Args:
            data_dir (Path): Path to the directory containing the annotated frames.
                The directory should contain frames: frame_*.jpg with corresponding masks mask_*.jpg.
            transform (callable): A function that transforms the raw images for the model.
            random_crop_to_square (bool): Whether to randomly crop frames to square. Default is true.
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.indices = [p.stem.split("_")[1] for p in self.data_dir.glob("mask_*.png")]
        self.transform = transform
        self.random_crop_to_square = random_crop_to_square

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        frame_idx = self.indices[index]
        frame = cv2.imread(self.data_dir / f"frame_{frame_idx}.jpg")
        mask = cv2.imread(self.data_dir / f"mask_{frame_idx}.png", cv2.IMREAD_GRAYSCALE)

        if self.random_crop_to_square:
            frame, mask = random_square_crop(frame, mask)
        
        frame_transformed = self.transform(frame)
        mask_transformed = torch.Tensor(mask).bool().float()
        return frame_transformed, mask_transformed
    

if __name__ == "__main__":
    # The input dimensions the backbone has been compiled with
    # during training we use square crops
    input_size = (518, 518)
    # The batch size to initialize the backbone with, for training, we use 16 for training
    batch_size = 16
    # The gpu the backbone and model should run on (if multiple are available)
    device = 0

    backbone = load_backbone(input_size=input_size, batch_size=batch_size, device=device)
    dante = DANTE(backbone=backbone).to(backbone.device)

    data_dir = "assets/training_data"
    dataset = TraversabilityDataset(data_dir=data_dir, transform=backbone.transform_cpu)

    # The training should run for 128 steps, therefore the dataset is repeated
    # Now it is only necessary to iterate through the dataloader once, which is much quicker
    num_epochs = 128
    dataloader = DataLoader(ConcatDataset([dataset]*(num_epochs*16//len(dataset))), batch_size=batch_size, shuffle=True, num_workers=16)

    # Initialize the trainer and start training
    trainer = L.Trainer(
        max_epochs=1, 
        enable_progress_bar=True,
        devices=[device],
        logger=False,
        enable_checkpointing=False
    )
    trainer.fit(
        model=dante, 
        train_dataloaders=dataloader, 
    )
    # Save the trained linear layer
    dante = dante.to("cpu")
    torch.save(dante.linear.state_dict(), "trained_decoder.pth")