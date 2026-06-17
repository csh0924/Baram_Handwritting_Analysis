from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import FeaturePyramidNetwork

from Analysis_Module.common_paths import find_project_root, get_handwriting_dir
from Analysis_Module.io_unicode_cv import imread_unicode, imwrite_unicode



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
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=256)

        self.head = nn.Sequential(
            nn.Conv2d(256 * 4, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape

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


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def imagenet_normalize_chw(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)[:, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device)[:, None, None]
    return (x - mean) / std


def assert_saved_matches_backbone(weight_path: Path, backbone: str):
    sd = torch.load(str(weight_path), map_location="cpu")
    has_conv3 = any(".conv3.weight" in k for k in sd.keys())

    if backbone in ("resnet18", "resnet34"):
        if has_conv3:
            raise RuntimeError(
                f"[BACKBONE MISMATCH] backbone={backbone} expects BasicBlock, "
                f"but Bottleneck ckpt seems saved: {weight_path}"
            )
    elif backbone in ("resnet50", "resnext50"):
        if not has_conv3:
            raise RuntimeError(
                f"[BACKBONE MISMATCH] backbone={backbone} expects Bottleneck, "
                f"but BasicBlock ckpt seems saved: {weight_path}"
            )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")


def ensure_out_dirs(out_root: Path) -> None:
    out_root = Path(out_root)
    (out_root / "labels_png").mkdir(parents=True, exist_ok=True)
    (out_root / "labels_npy").mkdir(parents=True, exist_ok=True)


def save_label_raw(out_root: Path, stem: str, pred_mask: np.ndarray) -> None:
    out_root = Path(out_root)
    ensure_out_dirs(out_root)
    m = pred_mask.astype(np.uint8)
    imwrite_unicode(out_root / "labels_png" / f"{stem}.png", m)
    np.save(str(out_root / "labels_npy" / f"{stem}.npy"), m)


def hangul_char_to_type(ch: str) -> str:
    """
    FPN weight 선택을 위한 문자 타입.
    (기존 로직 유지)
    """
    if len(ch) != 1:
        raise ValueError("Input must be a single character.")

    code = ord(ch)
    if code < 0xAC00 or code > 0xD7A3:
        raise ValueError(f"Non-Hangul syllable: {ch}")

    s_index = code - 0xAC00
    jong_index = s_index % 28
    jung_index = (s_index // 28) % 21
    has_jong = (jong_index != 0)

    horizontal = {8, 12, 13, 17, 18}
    vertical   = {0, 1, 2, 3, 4, 5, 6, 7, 20}

    if jung_index in horizontal:
        shape = "horizontal"
    elif jung_index in vertical:
        shape = "vertical"
    else:
        shape = "complex"

    return f"{shape}_{'jong' if has_jong else 'no_jong'}"


def build_default_weight_map(project_root: Path, backbone: str) -> Dict[str, Path]:
    wdir = project_root / "Analysis_Module" / "fpn_weights"
    if not wdir.exists():
        raise FileNotFoundError(f"fpn_weights dir not found: {wdir}")

    suffix = backbone
    weight_map = {
        "complex_jong":        wdir / f"fpn_complex_jong_{suffix}.pth",
        "complex_no_jong":     wdir / f"fpn_complex_no_jong_{suffix}.pth",
        "horizontal_jong":     wdir / f"fpn_horizontal_jong_{suffix}.pth",
        "horizontal_no_jong":  wdir / f"fpn_horizontal_no_jong_{suffix}.pth",
        "vertical_jong":       wdir / f"fpn_vertical_jong_{suffix}.pth",
        "vertical_no_jong":    wdir / f"fpn_vertical_no_jong_{suffix}.pth",
    }

    missing = [k for k, p in weight_map.items() if not p.exists()]
    if missing:
        msg = "\n".join([f"  - {k}: {weight_map[k]}" for k in missing])
        raise FileNotFoundError(f"Missing FPN weights:\n{msg}")

    return weight_map


def _list_images(folder: Path, exts: Tuple[str, ...]) -> List[Path]:
    folder = Path(folder)
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: p.name)
    return files



@dataclass
class FPNSegmentConfig:
    backbone: str = "resnet34"
    pretrained_backbone: bool = False
    force_cpu: bool = False
    check_backbone_match: bool = True

    size: Tuple[int, int] = (256, 256)  
    file_exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


@dataclass
class FPNSegmentResult:
    handwriting_id: str
    sentence: str
    char_dir: str
    out_dir: str
    manifest_path: str
    processed: int
    used_pairs: int
    warn_mismatch: bool
    used_fonts: bool
    config: dict


@torch.no_grad()
def run_fpn_segment(
    handwriting_id: str,
    original_text: str,
    config: Optional[FPNSegmentConfig] = None,
    project_root: Optional[Union[str, Path]] = None,
    *,
    char_dir: Optional[Union[str, Path]] = None,
    out_dir: Optional[Union[str, Path]] = None,
    fonts: bool = False,
) -> FPNSegmentResult:
    if config is None:
        config = FPNSegmentConfig()

    if project_root is None:
        project_root = find_project_root(marker_dir="Analysis_Module")
    else:
        project_root = Path(project_root)

    handwriting_dir = get_handwriting_dir(project_root, handwriting_id)
    weight_map = build_default_weight_map(project_root, backbone=config.backbone)

    if char_dir is None:
        char_dir = handwriting_dir / ("printed_chars" if fonts else "char_cropped")
    else:
        char_dir = Path(char_dir)

    if out_dir is None:
        out_dir = handwriting_dir / ("printed_segments" if fonts else "segments")
    else:
        out_dir = Path(out_dir)

    if not Path(char_dir).exists():
        raise FileNotFoundError(f"char_dir not found: {char_dir}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ensure_out_dirs(Path(out_dir))

    manifest_path = Path(out_dir) / "manifest.json"
    log_path = Path(out_dir) / "log.txt"

    files = _list_images(Path(char_dir), config.file_exts)
    if len(files) == 0:
        raise RuntimeError(f"No image files found in: {char_dir}")

    chars = [c for c in original_text if c != " "]
    warn_mismatch = False
    if len(chars) != len(files):
        warn_mismatch = True
        n = min(len(chars), len(files))
        chars = chars[:n]
        files = files[:n]

    device = get_device(force_cpu=config.force_cpu)
    model_cache: Dict[str, torch.nn.Module] = {}
    processed = 0

    for ch, img_path in zip(chars, files):
        type_name = hangul_char_to_type(ch)
        wpath = Path(weight_map[type_name])

        if config.check_backbone_match:
            assert_saved_matches_backbone(wpath, config.backbone)

        if type_name not in model_cache:
            model = ResNetFPN(
                num_classes=4,
                backbone=config.backbone,
                pretrained=config.pretrained_backbone
            ).to(device)
            state = torch.load(str(wpath), map_location=device)
            model.load_state_dict(state)
            model.eval()
            model_cache[type_name] = model

        model = model_cache[type_name]

        img_bgr = imread_unicode(str(img_path), flags=cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (config.size[1], config.size[0]), interpolation=cv2.INTER_LINEAR)

        x = img_rgb.astype(np.float32) / 255.0
        x = torch.from_numpy(x).permute(2, 0, 1).to(device)
        x = imagenet_normalize_chw(x).unsqueeze(0)

        logits = model(x)
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

        stem = img_path.stem
        save_label_raw(Path(out_dir), stem, pred)
        processed += 1

    result = FPNSegmentResult(
        handwriting_id=handwriting_id,
        sentence=original_text,
        char_dir=str(char_dir),
        out_dir=str(out_dir),
        manifest_path=str(manifest_path),
        processed=processed,
        used_pairs=len(chars),
        warn_mismatch=warn_mismatch,
        used_fonts=bool(fonts),
        config=asdict(config),
    )

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("[OK] fpn_segment\n")
        f.write(f"  handwriting_id: {handwriting_id}\n")
        f.write(f"  fonts_mode    : {fonts}\n")
        f.write(f"  char_dir      : {char_dir}\n")
        f.write(f"  out_dir       : {out_dir}\n")
        f.write(f"  processed     : {processed}\n")
        f.write(f"  used_pairs    : {len(chars)}\n")
        if warn_mismatch:
            f.write("  [WARN] mismatch between text chars (no-space) and files. used min(n).\n")

    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Stage: fpn_segment (cho/jung/jong via FPN)")
    ap.add_argument("--handwriting_id", required=True)
    ap.add_argument("--original_text", required=True)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--fonts", action="store_true", help="use printed_chars as input, save to printed_segments")
    args = ap.parse_args()

    cfg = FPNSegmentConfig(force_cpu=args.cpu)
    run_fpn_segment(
        handwriting_id=args.handwriting_id,
        original_text=args.original_text,
        config=cfg,
        fonts=args.fonts,
    )
