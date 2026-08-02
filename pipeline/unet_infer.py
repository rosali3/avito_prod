"""
UNet-simple (внешняя модель, семантическая сегментация) — альтернатива
RF-DETR в этом пайплайне. Чекпоинт с HuggingFace Hub
(nabiullina-dstu/avito-floorplan-checkpoints, unet_baseline/best_model.pt).

Архитектура и предобработка — точная копия claude_instseg_compare/models/
unet_baseline/infer_and_eval.py (ImageNet mean/std, resize 256x256),
не редактируем логику модели, только инференс.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

HF_REPO = "nabiullina-dstu/avito-floorplan-checkpoints"
HF_FILENAME = "unet_baseline/best_model.pt"
TRAIN_RESOLUTION = 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, base=64):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 8, base * 16)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.out_conv = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


def _preprocess(image_bgr: np.ndarray) -> torch.Tensor:
    img = cv2.resize(image_bgr, (TRAIN_RESOLUTION, TRAIN_RESOLUTION), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()


def load_model():
    global _model
    if _model is None:
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME)
        ckpt = torch.load(ckpt_path, map_location=_device, weights_only=False)
        _model = UNet(num_classes=ckpt["num_classes"], base=ckpt["base_filters"]).to(_device)
        _model.load_state_dict(ckpt["model_state"])
        _model.eval()
    return _model


def run_unet(image_bgr: np.ndarray) -> np.ndarray:
    """Возвращает сырую semantic-маску [H,W], id 0-7 (0=фон), в исходном
    разрешении входного изображения (upsampled от 256x256, nearest)."""
    model = load_model()
    h, w = image_bgr.shape[:2]
    x = _preprocess(image_bgr).to(_device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    pred_small = probs.argmax(0).astype(np.uint8)
    return cv2.resize(pred_small, (w, h), interpolation=cv2.INTER_NEAREST)
