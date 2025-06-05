import numpy as np
import torch.nn.functional as F
import torch

from trt_model import load_backbone, TensorRTModel
from torch import nn
import lightning as L


class DANTE(L.LightningModule):
    """
    The DANTE model (Depth ANything Based Traversability Estimation).
    """
    def __init__(self, backbone: TensorRTModel, backbone_type="s"):
        """
        Args:
            backbone (TensorRTModel): The compiled ViT backbone.
            backbone_type (str): The ViT type (s, b, l).
        """
        super(DANTE, self).__init__()
        input_dim = dict(
            s=384,
            b=768,
            l=1024,
            g=1536,
        )[backbone_type]
        self.linear = nn.Conv2d(in_channels=input_dim, out_channels=1, kernel_size=1)

        self.backbone = backbone
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, frames):
        features, cls_token = self.backbone.infer(frames)
        out = self.linear(features).squeeze(-3)
        if not self.training:
            out = torch.sigmoid(out)
        return out
        
    def training_step(self, batch):
        frames, masks = batch
        out = self(frames)
        masks_resized = F.interpolate(masks.unsqueeze(1).float(), size=out.shape[-2:], mode="nearest-exact").squeeze(1)

        loss = self.loss_fn(out, masks_resized)
        self.log("loss", loss, prog_bar=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)

    def infer_image(self, image: np.ndarray) -> np.ndarray:
        """
        Run the inference on an RGB image.

        Args:
            image (np.ndarray): The RGB image of shape (H, W, 3).

        Returns:
            (np.ndarray) The predicted mask of shape (H, W).
        """
        self.eval()
        image_transformed = self.backbone.transform(image)
        with torch.no_grad():
            mask_small = self(image_transformed).unsqueeze(0)
        mask_large = F.interpolate(
            mask_small, 
            size=image.shape[:2], 
            mode="bilinear", 
            align_corners=False
        ).squeeze(0).squeeze(0).cpu().numpy()
        return mask_large


if __name__ == "__main__":
    # The input dimensions the model backbone has been compiled with
    input_size = (518, 924)
    # The batch size to initialize the backbone with
    batch_size = 1
    # The gpu the backbone and model should run on (if multiple are available)
    device = 0

    backbone = load_backbone(input_size=input_size, batch_size=batch_size, device=device)
    dante = DANTE(backbone=backbone).to(backbone.device)

    image = np.random.randint(0, 256, size=(720, 1280, 3), dtype=np.uint8)
    mask = dante.infer_image(image)
    print(mask.shape)