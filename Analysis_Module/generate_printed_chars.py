from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from Analysis_Module.common_paths import find_project_root, get_handwriting_dir
from Analysis_Module.io_unicode_cv import imwrite_unicode

def _fit_font_size(
    text: str,
    font_path: str,
    canvas_size: Tuple[int, int],
    margin: int,
    max_size: int,
    min_size: int,
) -> ImageFont.FreeTypeFont:
    W, H = canvas_size
    target_w = max(1, W - 2 * margin)
    target_h = max(1, H - 2 * margin)

    dummy_img = Image.new("L", (W, H), 0)
    dummy_draw = ImageDraw.Draw(dummy_img)

    best_font = ImageFont.truetype(font_path, size=min_size)
    lo, hi = min_size, max_size

    while lo <= hi:
        mid = (lo + hi) // 2
        font = ImageFont.truetype(font_path, size=mid)

        bbox = dummy_draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        if text_w <= target_w and text_h <= target_h:
            best_font = font
            lo = mid + 1
        else:
            hi = mid - 1

    return best_font


def _render_single(
    text: str,
    font_path: str,
    image_size: Tuple[int, int],
    margin: int,
    max_font_size: int,
    min_font_size: int,
) -> Image.Image:
    W, H = image_size
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)

    font = _fit_font_size(
        text=text,
        font_path=font_path,
        canvas_size=(W, H),
        margin=margin,
        max_size=max_font_size,
        min_size=min_font_size,
    )

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = (W - text_w) // 2 - bbox[0]
    y = (H - text_h) // 2 - bbox[1]

    draw.text((x, y), text, fill=255, font=font)
    return img


def _resolve_font_path(project_root: Path, font_name: Optional[str]) -> Path:
    fonts_dir = project_root / "Analysis_Module" / "fonts"
    if not fonts_dir.exists():
        raise FileNotFoundError(f"fonts dir not found: {fonts_dir}")

    if font_name is None:
        cand = sorted([p for p in fonts_dir.iterdir() if p.is_file() and p.suffix.lower() in (".ttf", ".otf")])
        if not cand:
            raise FileNotFoundError(f"No .ttf/.otf fonts found in: {fonts_dir}")
        return cand[0]

    font_path = fonts_dir / font_name
    if not font_path.exists():
        raise FileNotFoundError(f"Font not found: {font_path}")
    return font_path


def _list_images(folder: Path, exts: Tuple[str, ...]) -> List[Path]:
    folder = Path(folder)
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: p.name)
    return files



@dataclass
class PrintedCharsConfig:
    image_size: Tuple[int, int] = (256, 256)
    margin: int = 18
    overwrite: bool = True
    max_font_size: int = 220
    min_font_size: int = 30

    skip_spaces: bool = True

    font_name: Optional[str] = None 

    file_exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


@dataclass
class PrintedCharsResult:
    handwriting_id: str
    sentence: str
    font_path: str
    ref_char_cropped_dir: str
    out_dir: str
    manifest_path: str
    saved_count: int
    used_chars: int
    used_filenames: int
    warn_spaces_dropped: bool
    warn_count_mismatch: bool
    config: dict



def run_printed_chars(
    handwriting_id: str,
    original_text: str,
    config: Optional[PrintedCharsConfig] = None,
    project_root: Optional[Union[str, Path]] = None,
    out_dir: Optional[Union[str, Path]] = None,
) -> PrintedCharsResult:
    if config is None:
        config = PrintedCharsConfig()

    project_root = Path(project_root) if project_root is not None else find_project_root(marker_dir="Analysis_Module")
    font_path = _resolve_font_path(project_root, config.font_name)
    handwriting_dir = get_handwriting_dir(project_root, handwriting_id)

    ref_dir = handwriting_dir / "char_cropped"
    if not ref_dir.exists():
        raise FileNotFoundError(f"char_cropped dir not found: {ref_dir}")

    if out_dir is None:
        out_dir = handwriting_dir / "printed_chars"
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.json"
    log_path = out_dir / "log.txt"

    chars: List[str] = []
    warn_spaces_dropped = False
    for ch in original_text:
        if config.skip_spaces and ch == " ":
            warn_spaces_dropped = True
            continue
        chars.append(ch)

    ref_files = _list_images(ref_dir, config.file_exts)
    if len(ref_files) == 0:
        raise RuntimeError(f"No images in: {ref_dir}")

    warn_count_mismatch = (len(chars) != len(ref_files))
    n = min(len(chars), len(ref_files))

    chars = chars[:n]
    ref_files = ref_files[:n]
    filenames = [p.name for p in ref_files] 

    saved_count = 0
    for ch, fn in zip(chars, filenames):
        img_pil = _render_single(
            text=ch,
            font_path=str(font_path),
            image_size=config.image_size,
            margin=config.margin,
            max_font_size=config.max_font_size,
            min_font_size=config.min_font_size,
        )

        out_path = out_dir / fn
        if out_path.exists() and not config.overwrite:
            continue

        arr = np.array(img_pil, dtype=np.uint8)
        ok = imwrite_unicode(out_path, arr)
        if not ok:
            raise RuntimeError(f"Failed to write: {out_path}")

        saved_count += 1

    result = PrintedCharsResult(
        handwriting_id=handwriting_id,
        sentence=original_text,
        font_path=str(font_path),
        ref_char_cropped_dir=str(ref_dir),
        out_dir=str(out_dir),
        manifest_path=str(manifest_path),
        saved_count=saved_count,
        used_chars=len(chars),
        used_filenames=len(filenames),
        warn_spaces_dropped=warn_spaces_dropped,
        warn_count_mismatch=warn_count_mismatch,
        config=asdict(config),
    )

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("[OK] printed_chars\n")
        f.write(f"  handwriting_id     : {handwriting_id}\n")
        f.write(f"  font              : {font_path}\n")
        f.write(f"  ref_char_cropped  : {ref_dir}\n")
        f.write(f"  out_dir           : {out_dir}\n")
        f.write(f"  saved_count       : {saved_count}\n")
        f.write(f"  used_chars        : {len(chars)}\n")
        f.write(f"  used_filenames    : {len(filenames)}\n")
        if warn_spaces_dropped:
            f.write("  [WARN] spaces were dropped (skip_spaces=True)\n")
        if warn_count_mismatch:
            f.write("  [WARN] count mismatch between chars(no-space) and char_cropped files. used min(n).\n")

    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Stage: printed_chars (match char_cropped filenames always)")
    ap.add_argument("--handwriting_id", required=True)
    ap.add_argument("--original_text", required=True)
    ap.add_argument("--font_name", default=None)
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing printed images")
    args = ap.parse_args()

    cfg = PrintedCharsConfig(
        font_name=args.font_name,
        overwrite=bool(args.overwrite),
    )
    out = run_printed_chars(
        handwriting_id=args.handwriting_id,
        original_text=args.original_text,
        config=cfg,
    )
    print(out)