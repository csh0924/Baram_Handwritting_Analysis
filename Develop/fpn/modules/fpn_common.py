#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.ops import FeaturePyramidNetwork

def setup_project_path() -> Path:
    current = Path.cwd()
    while current != current.parent and not (current / "fpn").exists():
        current = current.parent
    if not (current / "fpn").exists():
        raise RuntimeError("Could not find project_root containing 'fpn' directory.")
    return current


# In[3]:


def imwrite_unicode(path, img):
    path = str(path)
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    try:
        path = str(path)
        with open(path, "rb") as f:
            data = f.read()
        img_array = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(img_array, flags)
        return img
    except Exception as e:
        print("[imread_unicode ERROR]", e)
        return None


# In[5]:


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



# In[9]:


class ResNetFPN(nn.Module):
    def __init__(self, num_classes: int = 4, backbone: str = "resnet34", pretrained: bool = False):
        super().__init__()

        if backbone == "resnet34":
            base = torchvision.models.resnet34(
                weights=torchvision.models.ResNet34_Weights.DEFAULT if pretrained else None
            )
            in_channels_list = [64, 128, 256, 512]

        elif backbone == "resnet18":
            base = torchvision.models.resnet18(
                weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            in_channels_list = [64, 128, 256, 512]

        elif backbone == "resnet50":
            base = torchvision.models.resnet50(
                weights=torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
            )
            in_channels_list = [256, 512, 1024, 2048]

        elif backbone == "resnext50":
            base = torchvision.models.resnext50_32x4d(
                weights=torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
            )
            in_channels_list = [256, 512, 1024, 2048]

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)

        self.layer1 = base.layer1  # C2
        self.layer2 = base.layer2  # C3
        self.layer3 = base.layer3  # C4
        self.layer4 = base.layer4  # C5

        self.fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=256)

        self.head = nn.Sequential(
            nn.Conv2d(256 * 4, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, num_classes, 1)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        feats = self.fpn({"c2": c2, "c3": c3, "c4": c4, "c5": c5})

        p2 = F.interpolate(feats["c2"], size=(H, W), mode="bilinear", align_corners=False)
        p3 = F.interpolate(feats["c3"], size=(H, W), mode="bilinear", align_corners=False)
        p4 = F.interpolate(feats["c4"], size=(H, W), mode="bilinear", align_corners=False)
        p5 = F.interpolate(feats["c5"], size=(H, W), mode="bilinear", align_corners=False)

        fused = torch.cat([p2, p3, p4, p5], dim=1)
        return self.head(fused)


# In[11]:


def imagenet_normalize_chw(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)[:, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device)[:, None, None]
    return (x - mean) / std


# In[13]:


def assert_saved_matches_backbone(weight_path: Path, backbone: str):
    sd = torch.load(weight_path, map_location="cpu")
    has_conv3 = any(".conv3.weight" in k for k in sd.keys())

    if backbone in ("resnet18", "resnet34"):
        if has_conv3:
            raise RuntimeError(
                f"[BACKBONE MISMATCH] backbone={backbone} expects BasicBlock, "
                f"but Bottleneck ckpt saved: {weight_path}"
            )
    elif backbone in ("resnet50", "resnext50"):
        if not has_conv3:
            raise RuntimeError(
                f"[BACKBONE MISMATCH] backbone={backbone} expects Bottleneck, "
                f"but BasicBlock ckpt saved: {weight_path}"
            )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")


# In[15]:


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    palette = np.array([
        [0,   0,   0  ],   # 0 background
        [255, 0,   0  ],   # 1 cho
        [0,   255, 0  ],   # 2 jung
        [0,   0,   255],   # 3 jong
    ], dtype=np.uint8)
    return palette[np.clip(mask, 0, 3)]


def overlay(img_rgb: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = (1 - alpha) * img_rgb.astype(np.float32) + alpha * mask_rgb.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


# In[17]:


def hangul_char_to_type(ch: str) -> str:
    if len(ch) != 1:
        raise ValueError("Input must be a single character.")

    code = ord(ch)
    if code < 0xAC00 or code > 0xD7A3:
        raise ValueError(f"Non-Hangul syllable: {ch}")

    s_index = code - 0xAC00
    jong_index = s_index % 28
    jung_index = (s_index // 28) % 21

    has_jong = (jong_index != 0)

    horizontal = {8, 12, 13, 17, 18}                 # ㅗ,ㅛ,ㅜ,ㅠ,ㅡ
    vertical   = {0, 1, 2, 3, 4, 5, 6, 7, 20}        # ㅏ,ㅐ,ㅑ,ㅒ,ㅓ,ㅔ,ㅕ,ㅖ,ㅣ
    complex_   = {9, 10, 11, 14, 15, 16, 19}         # ㅘ,ㅙ,ㅚ,ㅝ,ㅞ,ㅟ,ㅢ

    if jung_index in horizontal:
        shape = "horizontal"
    elif jung_index in vertical:
        shape = "vertical"
    else:
        shape = "complex"

    return f"{shape}_{'jong' if has_jong else 'no_jong'}"

