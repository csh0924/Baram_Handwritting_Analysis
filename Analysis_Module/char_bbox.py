from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, Union, Dict

import cv2
import numpy as np

from Analysis_Module.common_paths import find_project_root, get_handwriting_dir
from Analysis_Module.io_unicode_cv import imread_unicode, imwrite_unicode


def extract_bbox_xyxy(img_bgr_or_gray: np.ndarray, threshold: int = 0) -> Optional[Tuple[int, int, int, int]]:
    if img_bgr_or_gray is None:
        return None

    if img_bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr_or_gray.copy()

    ys, xs = np.where(gray > threshold)
    if xs.size == 0:
        return None

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1, y1)


def crop_square_including_bbox(
    img: np.ndarray,
    xyxy: Tuple[int, int, int, int],
    pad: int = 0,
    pad_value: int = 0,
) -> np.ndarray:
    x0, y0, x1, y1 = xyxy
    h, w = img.shape[:2]

    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    side = max(bw, bh) + 2 * pad
    side = max(1, int(side))

    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    half = side / 2.0

    sx0 = int(math.floor(cx - half))
    sy0 = int(math.floor(cy - half))
    sx1 = int(math.ceil(cx + half - 1))
    sy1 = int(math.ceil(cy + half - 1))

    left = max(0, -sx0)
    top = max(0, -sy0)
    right = max(0, sx1 - (w - 1))
    bottom = max(0, sy1 - (h - 1))

    if any(v > 0 for v in (left, top, right, bottom)):
        if img.ndim == 2:
            img_pad = cv2.copyMakeBorder(
                img, top, bottom, left, right,
                borderType=cv2.BORDER_CONSTANT,
                value=pad_value,
            )
        else:
            img_pad = cv2.copyMakeBorder(
                img, top, bottom, left, right,
                borderType=cv2.BORDER_CONSTANT,
                value=(pad_value, pad_value, pad_value),
            )
        sx0 += left; sx1 += left
        sy0 += top;  sy1 += top
    else:
        img_pad = img

    return img_pad[sy0:sy1 + 1, sx0:sx1 + 1]


def center_on_canvas(img_small: np.ndarray, canvas_size: int = 256, pad_value: int = 0) -> np.ndarray:
    h, w = img_small.shape[:2]
    if h > canvas_size or w > canvas_size:
        raise ValueError(f"Input larger than canvas: {h}x{w} > {canvas_size}x{canvas_size}")

    if img_small.ndim == 2:
        canvas = np.full((canvas_size, canvas_size), pad_value, dtype=img_small.dtype)
    else:
        canvas = np.full((canvas_size, canvas_size, img_small.shape[2]), pad_value, dtype=img_small.dtype)

    y0 = (canvas_size - h) // 2
    x0 = (canvas_size - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = img_small
    return canvas


def save_centered_inner_in_256(
    img: np.ndarray,
    xyxy: Tuple[int, int, int, int],
    out_path: Path,
    crop_pad: int = 4,
    inner_size: int = 192,
    canvas_size: int = 256,
    pad_value: int = 0,
) -> bool:
    crop = crop_square_including_bbox(img, xyxy, pad=crop_pad, pad_value=pad_value)
    if crop.size == 0:
        return False

    interp = cv2.INTER_AREA if crop.shape[0] > inner_size else cv2.INTER_CUBIC
    inner = cv2.resize(crop, (inner_size, inner_size), interpolation=interp)

    canvas = center_on_canvas(inner, canvas_size=canvas_size, pad_value=pad_value)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return imwrite_unicode(out_path, canvas)


@dataclass
class CharBBoxConfig:
    threshold: int = 0
    read_flags: int = cv2.IMREAD_UNCHANGED
    crop_pad: int = 4
    inner_size: int = 192
    canvas_size: int = 256
    pad_value: int = 0


@dataclass
class CharBBoxResult:
    handwriting_id: str
    chars_dir: str
    bbox_txt: str
    cropped_dir: str
    manifest_path: str
    processed: int
    skipped_no_ink: int
    config: dict


def run_char_bbox(
    handwriting_id: str,
    config: Optional[CharBBoxConfig] = None,
    project_root: Optional[Union[Path, str]] = None,
    chars_dir: Optional[Union[Path, str]] = None,
    roi_map: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
) -> CharBBoxResult:
    if config is not None and not isinstance(config, CharBBoxConfig):
        raise TypeError(
            f"config must be CharBBoxConfig or None, got {type(config)}. "
        )
    if config is None:
        config = CharBBoxConfig()

    if project_root is None:
        project_root = find_project_root(marker_dir="Analysis_Module")
    else:
        project_root = Path(project_root)

    handwriting_dir = get_handwriting_dir(project_root, handwriting_id)

    if chars_dir is None:
        chars_dir = handwriting_dir / "chars"
    else:
        chars_dir = Path(chars_dir)

    if not chars_dir.exists():
        raise FileNotFoundError(f"chars dir not found: {chars_dir}")

    if roi_map is None:
        roi_map = {}

    out_txt_dir = handwriting_dir / "char_bbox"
    out_crop_dir = handwriting_dir / "char_cropped"
    out_txt_dir.mkdir(parents=True, exist_ok=True)
    out_crop_dir.mkdir(parents=True, exist_ok=True)

    bbox_txt = out_txt_dir / "charbbox.txt"
    manifest_path = out_txt_dir / "manifest.json"
    log_path = out_txt_dir / "log.txt"

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    images = sorted(
        [p for p in chars_dir.iterdir() if p.is_file() and p.suffix.lower() in exts],
        key=lambda p: p.name
    )

    processed = 0
    skipped_no_ink = 0

    with open(bbox_txt, "w", encoding="utf-8") as f:
        f.write(f"# handwriting_id: {handwriting_id}\n")
        f.write("-" * 60 + "\n")

        for p in images:
            img = imread_unicode(str(p), flags=config.read_flags)
            if img is None:
                f.write(f"file={p.name}\n")
                f.write("  INVALID (failed to read)\n")
                f.write("-" * 60 + "\n")
                skipped_no_ink += 1
                continue

            xyxy_local = extract_bbox_xyxy(img, threshold=config.threshold)

            f.write(f"file={p.name}\n")
            if xyxy_local is None:
                f.write("  INVALID (no ink pixels)\n")
                f.write("-" * 60 + "\n")
                skipped_no_ink += 1
                continue

            roi_xyxy = roi_map.get(p.name)
            if roi_xyxy is not None:
                rx0, ry0, rx1, ry1 = roi_xyxy
                x0, y0, x1, y1 = xyxy_local
                xyxy_to_write = (rx0 + x0, ry0 + y0, rx0 + x1, ry0 + y1)
            else:
                xyxy_to_write = xyxy_local

            x0, y0, x1, y1 = xyxy_to_write
            f.write(f"  xyxy=({x0},{y0},{x1},{y1})\n")
            f.write("-" * 60 + "\n")

            out_img_path = out_crop_dir / p.name
            ok = save_centered_inner_in_256(
                img=img,
                xyxy=xyxy_local,
                out_path=out_img_path,
                crop_pad=config.crop_pad,
                inner_size=config.inner_size,
                canvas_size=config.canvas_size,
                pad_value=config.pad_value,
            )
            if ok:
                processed += 1

    result = CharBBoxResult(
        handwriting_id=handwriting_id,
        chars_dir=str(chars_dir),
        bbox_txt=str(bbox_txt),
        cropped_dir=str(out_crop_dir),
        manifest_path=str(manifest_path),
        processed=processed,
        skipped_no_ink=skipped_no_ink,
        config=asdict(config),
    )

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("[OK] char_bbox\n")
        f.write(f"  chars_dir: {chars_dir}\n")
        f.write(f"  bbox_txt : {bbox_txt}\n")
        f.write(f"  cropped  : {out_crop_dir}\n")
        f.write(f"  processed: {processed}\n")
        f.write(f"  skipped  : {skipped_no_ink}\n")

    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Stage: char_bbox (bbox txt + inner-in-256 crop)")
    ap.add_argument("--handwriting_id", required=True)
    args = ap.parse_args()

    run_char_bbox(handwriting_id=args.handwriting_id)