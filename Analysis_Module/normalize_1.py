from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
from skimage.morphology import skeletonize, disk

from Analysis_Module.common_paths import find_project_root, get_handwriting_dir
from Analysis_Module.io_unicode_cv import imread_unicode, imwrite_unicode


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


@dataclass
class Normalize1Config:
    target_stroke_px: int = 10
    binarize_method: str = "otsu"  
    invert: bool = True
    open_kernel: int = 3
    read_flags: int = cv2.IMREAD_COLOR


@dataclass
class Normalize1Result:
    handwriting_id: str
    input_image_path: str
    out_dir: str
    normalized_image_path: str
    manifest_path: str
    config: dict


def _pick_single_image(handwriting_dir: Path) -> Path:
    files = [p for p in handwriting_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if len(files) == 0:
        raise FileNotFoundError(f"이미지 파일을 찾지 못했습니다: {handwriting_dir}")
    if len(files) > 1:
        files_sorted = sorted(files, key=lambda p: p.name)
        raise RuntimeError(
            f"handwriting_dir에 이미지가 2장 이상 존재합니다(계약 위반). "
            f"발견: {len(files)}개, 예: {files_sorted[:3]}"
        )
    return files[0]


def normalize_stroke_width(
    img: np.ndarray,
    target_stroke_px: int,
    binarize_method: str = "otsu",
    invert: bool = True,
    open_kernel: int = 3,
) -> np.ndarray:
    if target_stroke_px <= 0:
        raise ValueError("target_stroke_px는 1 이상의 정수여야 합니다.")
    if img is None:
        raise ValueError("입력 이미지가 None입니다.")

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    if binarize_method == "adaptive":
        bin_img = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 5
        )
    else:
        _, bin_img = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    if invert:
        bin_img = 255 - bin_img

    if open_kernel and open_kernel >= 3:
        k = np.ones((open_kernel, open_kernel), np.uint8)
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, k)

    binary_bool = bin_img > 0
    skel_bool = skeletonize(binary_bool)

    radius = max(1, target_stroke_px // 2)
    selem = disk(radius)
    rendered = cv2.dilate(skel_bool.astype(np.uint8), selem, iterations=1)

    normalized_mask = (rendered > 0).astype(np.uint8) * 255
    return normalized_mask


def run_normalize_1(
    handwriting_id: str,
    config: Optional[Normalize1Config] = None,
    project_root: Optional[Path] = None,
) -> Normalize1Result:
    config = config or Normalize1Config()
    project_root = project_root or find_project_root(marker_dir="Analysis_Module")

    handwriting_dir = get_handwriting_dir(project_root, handwriting_id)
    if not handwriting_dir.exists():
        raise FileNotFoundError(f"handwriting_dir가 존재하지 않습니다: {handwriting_dir}")

    in_img_path = _pick_single_image(handwriting_dir)

    out_dir = handwriting_dir / "normalize_1"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_img_path = out_dir / "normalized.png"
    manifest_path = out_dir / "manifest.json"
    log_path = out_dir / "log.txt"

    img = imread_unicode(in_img_path, flags=config.read_flags)
    if img is None:
        raise RuntimeError(f"이미지를 읽지 못했습니다: {in_img_path}")

    norm_mask = normalize_stroke_width(
        img,
        target_stroke_px=config.target_stroke_px,
        binarize_method=config.binarize_method,
        invert=config.invert,
        open_kernel=config.open_kernel,
    )

    if not imwrite_unicode(out_img_path, norm_mask):
        raise RuntimeError(f"출력 저장 실패: {out_img_path}")

    result = Normalize1Result(
        handwriting_id=handwriting_id,
        input_image_path=str(in_img_path),
        out_dir=str(out_dir),
        normalized_image_path=str(out_img_path),
        manifest_path=str(manifest_path),
        config=asdict(config),
    )

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("[OK] normalize_1\n")
        f.write(f"  input : {in_img_path}\n")
        f.write(f"  output: {out_img_path}\n")
        f.write(f"  config: {asdict(config)}\n")

    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--handwriting_id", required=True)
    ap.add_argument("--target_stroke_px", type=int, default=15)
    ap.add_argument("--binarize_method", type=str, default="otsu", choices=["otsu", "adaptive"])
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--no_invert", action="store_true")
    ap.add_argument("--open_kernel", type=int, default=3)

    args = ap.parse_args()

    invert = True
    if args.no_invert:
        invert = False
    elif args.invert:
        invert = True

    cfg = Normalize1Config(
        target_stroke_px=args.target_stroke_px,
        binarize_method=args.binarize_method,
        invert=invert,
        open_kernel=args.open_kernel,
    )

    run_normalize_1(args.handwriting_id, cfg)
