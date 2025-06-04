from pathlib import Path
from torch.utils.data import Dataset
import cv2
import torch.nn.functional as F
import torch
import random

from tqdm import tqdm
from trt_model import TensorRTModel
from torch import nn
import lightning as L
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset


def random_square_crop(frame, mask):
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
    def __init__(self, data_dir: Path, transform, random_crop_to_square=True):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.indices = [p.stem.split("_")[1] for p in self.data_dir.glob("mask_*.png")]
        self.transform = transform
        self.random_crop_to_square = random_crop_to_square

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        frame_idx = self.indices[index]
        frame = cv2.imread(self.data_dir / f"{frame_idx}.png")
        mask = cv2.imread(self.data_dir / f"mask_{frame_idx}.png", cv2.IMREAD_GRAYSCALE)

        if self.random_crop_to_square:
            frame, mask = random_square_crop(frame, mask)
        
        frame_transformed = self.transform(frame).squeeze(0)
        mask_transformed = torch.Tensor(mask) / 255
        return frame_transformed, mask_transformed
    

class Linear(nn.Module):
    def __init__(self, backbone_type):
        super(Linear, self).__init__()
        input_dim = dict(
            s=384,
            b=768,
            l=1024,
            g=1536,
        )[backbone_type]
        self.linear = nn.Conv2d(input_dim, 1, 1)
        self.lr = 1e-3

    def forward(self, embedding):
        out = self.linear(embedding).squeeze(-3)
        if not self.training:
            out = torch.sigmoid(out)
        return out
    

def to_tensor_recursive(data, device=None):
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device=device)
    elif isinstance(data, list) or isinstance(data, tuple):
        return [to_tensor_recursive(item, device=device) for item in data]
    

class Model(L.LightningModule):
    def __init__(self, backbone, decoder_type="linear", backbone_type="s"):
        super(Model, self).__init__()
        self.decoder_type = decoder_type
        self.backbone_type = backbone_type
        self.model = dict(
            linear=Linear,
        )[self.decoder_type](self.backbone_type)
        self.backbone = backbone
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, frames):
        features, cls_token = to_tensor_recursive(self.backbone.infer(frames), self.device)
        return self.model(features)
    
    def training_step(self, batch, batch_idx):
        frames, masks = batch
        out = self(frames)
        masks_resized = F.interpolate(masks.unsqueeze(1).float(), size=out.shape[-2:]).squeeze(1)

        loss = self.loss_fn(out, masks_resized)
        self.log("loss", loss, prog_bar=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.model.lr)
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = self.model.state_dict()
        

def run_on_video(video_path, output_path, model, backbone_type, input_size, dt=1/3):
    _ = torch.zeros(0, device=model.device)
    engine_path = f"/export/home/nschackn/laboratory/weights/{backbone_type}_{input_size[0]}x{input_size[1]}.trt"
    model._backbone = TensorRTModel(engine_path, input_shape=(1, 3, *input_size))
    model.eval()

    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, int(cap.get(cv2.CAP_PROP_FPS) * dt))
    num_digits = len(str(total_frames))

    i = 0
    with tqdm(total=total_frames//stride, desc="Running inference") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if i % stride == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                predicted_mask = (predict_frame(model, frame) * 255).astype(np.uint8)
                cv2.imwrite(output_path / f"{i:0{num_digits}}.png", predicted_mask)
                pbar.update(1)

            i += 1



if __name__ == "__main__":
    input_size = (518, 518)
    backbone_type = "depth_anything_v2_s"
    num_epochs = 128
    batch_size = 16
    device = 0
    _ = torch.zeros(0, device=f"cuda:{device}")

    engine_path = f"/export/home/nschackn/laboratory/weights/{backbone_type}_{input_size[0]}x{input_size[1]}.trt"
    backbone = TensorRTModel(engine_path, input_shape=(batch_size, 3, *input_size))

    data_dir = "/export/home/nschackn/laboratory/to_annotate_ipa_train_8"
    dataset = TraversabilityDataset(data_dir, backbone.transform, True)

    dataloader = DataLoader(ConcatDataset([dataset]*(num_epochs*16//len(dataset))), batch_size=batch_size, shuffle=True, num_workers=16)
    model = Model(backbone)

    trainer = L.Trainer(
        max_epochs=1, 
        enable_progress_bar=True,
        devices=[device],
    )
    trainer.fit(
        model=model, 
        train_dataloaders=dataloader, 
    )

    run_on_video(
        video_path="/export/data/nschackn/dataset/data_ipa_train/input_video.mp4", 
        output_path="ipa_train_8",
        model=model, 
        backbone_type=backbone_type,
        input_size=(518, 924)
    )