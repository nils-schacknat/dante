import numpy as np
import torch.nn.functional as F
import torch

from trt_model import load_backbone
from torch import nn
import lightning as L


class DANTE(L.LightningModule):
    def __init__(self, backbone, backbone_type="s"):
        super(DANTE, self).__init__()
        input_dim = dict(
            s=384,
            b=768,
            l=1024,
            g=1536,
        )[backbone_type]
        self.linear = nn.Conv2d(in_channels=input_dim, out_channels=1, kernel_size=1)

        self._backbone = backbone
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, frames):
        features, cls_token = self._backbone.infer(frames)
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

    def infer_image(self, image):
        self.eval()
        image_transformed = self._backbone.transform(image)
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
    input_size = (518, 924)
    batch_size = 1
    device = 0

    backbone = load_backbone(input_size=input_size, batch_size=batch_size, device=device)
    dante = DANTE(backbone=backbone).to(backbone.device)

    image = np.random.randint(0, 256, size=(720, 1280, 3), dtype=np.uint8)
    mask = dante.infer_image(image)
    print(mask.shape)