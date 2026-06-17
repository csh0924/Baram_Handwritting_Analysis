from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np

from Analysis_Module.common_paths import find_project_root, get_handwriting_dir
from Analysis_Module.io_unicode_cv import imread_unicode, imwrite_unicode


def read_centers_char_format(center_txt: Path) -> List[Tuple[str, float, float]]:
    if not center_txt.exists():
        raise FileNotFoundError(f"center.txt not found: {center_txt}")

    out: List[Tuple[str, float, float]] = []
    with open(center_txt, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if ":" not in line:
                raise ValueError(f"Invalid center.txt line (missing ':') at {line_no}: {line}")
            ch, rest = line.split(":", 1)
            ch = ch.strip()
            rest = rest.strip()

            if "," not in rest:
                raise ValueError(f"Invalid center.txt line (missing ',') at {line_no}: {line}")

            xs, ys = rest.split(",", 1)
            x = float(xs.strip())
            y = float(ys.strip())
            out.append((ch, x, y))

    if not out:
        raise ValueError(f"center.txt is empty: {center_txt}")

    return out


def build_center_protect_mask(
    shape_hw: Tuple[int, int],
    centers: List[Tuple[str, float, float]],
    protect_radius_px: int = 18,
) -> np.ndarray:
    H, W = shape_hw
    mask = np.zeros((H, W), dtype=np.uint8)
    r = int(max(1, protect_radius_px))
    for _, x, y in centers:
        cx = int(round(x))
        cy = int(round(y))
        if 0 <= cx < W and 0 <= cy < H:
            cv2.circle(mask, (cx, cy), r, 1, thickness=-1)
    return mask


def remove_noise_by_craft_score_far_only(
    ink_white_on_black: np.ndarray,
    score_text: np.ndarray,       
    *,
    text_prob_thresh: float = 0.25,
    low_prob_thresh: float = 0.02,
    dist_thresh_px: float = 40.0,
    protect_dilate_r: int = 8,
    center_protect_mask: Optional[np.ndarray] = None, 
    connectivity: int = 8,
) -> np.ndarray:
    if ink_white_on_black.ndim != 2:
        raise ValueError("ink_white_on_black must be single-channel.")
    ink = (ink_white_on_black > 0).astype(np.uint8) * 255
    H, W = ink.shape[:2]

    s = score_text
    if s.ndim == 3:
        s = s.squeeze()
    if s.ndim != 2:
        raise ValueError("score_text must be 2D or squeezable to 2D.")
    s = s.astype(np.float32)

    if s.shape != (H, W):
        s = cv2.resize(s, (W, H), interpolation=cv2.INTER_LINEAR)

    text_region = (s >= float(text_prob_thresh)).astype(np.uint8)  

    if protect_dilate_r and protect_dilate_r > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * protect_dilate_r + 1, 2 * protect_dilate_r + 1)
        )
        text_region = cv2.dilate(text_region, k, iterations=1)

    if center_protect_mask is not None:
        if center_protect_mask.shape != (H, W):
            raise ValueError("center_protect_mask shape mismatch.")
        text_region = np.maximum(text_region, (center_protect_mask > 0).astype(np.uint8))

    inv = (1 - text_region).astype(np.uint8) * 255
    dist = cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=connectivity)

    out = ink.copy()
    for i in range(1, num):
        comp = (labels == i)

        comp_min_dist = float(dist[comp].min())
        if comp_min_dist < float(dist_thresh_px):
            continue

        comp_max_score = float(s[comp].max())
        if comp_max_score <= float(low_prob_thresh):
            out[comp] = 0

    return out


@dataclass
class Normalize2Config:
    text_prob_thresh: float = 0.25
    low_prob_thresh: float = 0.02
    dist_thresh_px: float = 40.0
    protect_dilate_r: int = 8
    connectivity: int = 8
    center_protect_radius_px: int = 18


@dataclass
class Normalize2Result:
    handwriting_id: str
    in_image_path: str
    score_path: str
    center_path: str
    out_dir: str
    out_image_path: str
    manifest_path: str
    config: dict


def run_normalize_2(
    handwriting_id: str,
    config: Optional[Normalize2Config] = None,
    project_root: Optional[Path] = None,
    in_image_path: Optional[str | Path] = None,
    score_path: Optional[str | Path] = None,
    center_txt_path: Optional[str | Path] = None,
) -> Normalize2Result:
    config = config or Normalize2Config()
    project_root = project_root or find_project_root(marker_dir="Analysis_Module")
    handwriting_dir = get_handwriting_dir(project_root, handwriting_id)

    if in_image_path is None:
        in_image_path = handwriting_dir / "normalize_1" / "normalized.png"
    else:
        in_image_path = Path(in_image_path)

    if score_path is None:
        score_path = handwriting_dir / "score" / "score.npy"
    else:
        score_path = Path(score_path)

    if center_txt_path is None:
        center_txt_path = handwriting_dir / "center" / "center.txt"
    else:
        center_txt_path = Path(center_txt_path)

    if not Path(in_image_path).exists():
        raise FileNotFoundError(f"normalize_1 image not found: {in_image_path}")
    if not Path(score_path).exists():
        raise FileNotFoundError(f"score.npy not found: {score_path}")
    if not Path(center_txt_path).exists():
        raise FileNotFoundError(f"center.txt not found: {center_txt_path}")

    out_dir = handwriting_dir / "normalize_2"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_image_path = out_dir / "normalized.png"
    manifest_path = out_dir / "manifest.json"
    log_path = out_dir / "log.txt"

    img = imread_unicode(str(in_image_path), flags=cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {in_image_path}")

    _, ink = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)

    score = np.load(str(score_path)).astype(np.float32)

    centers = read_centers_char_format(Path(center_txt_path))
    center_mask = build_center_protect_mask(
        shape_hw=ink.shape[:2],
        centers=centers,
        protect_radius_px=config.center_protect_radius_px,
    )

    cleaned = remove_noise_by_craft_score_far_only(
        ink_white_on_black=ink,
        score_text=score,
        text_prob_thresh=config.text_prob_thresh,
        low_prob_thresh=config.low_prob_thresh,
        dist_thresh_px=config.dist_thresh_px,
        protect_dilate_r=config.protect_dilate_r,
        center_protect_mask=center_mask,
        connectivity=config.connectivity,
    )

    if not imwrite_unicode(str(out_image_path), cleaned):
        raise RuntimeError(f"Failed to write image: {out_image_path}")

    result = Normalize2Result(
        handwriting_id=handwriting_id,
        in_image_path=str(in_image_path),
        score_path=str(score_path),
        center_path=str(center_txt_path),
        out_dir=str(out_dir),
        out_image_path=str(out_image_path),
        manifest_path=str(manifest_path),
        config=asdict(config),
    )

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("[OK] normalize_2\n")
        f.write(f"  in_image : {in_image_path}\n")
        f.write(f"  score    : {score_path}\n")
        f.write(f"  centers  : {center_txt_path}\n")
        f.write(f"  output   : {out_image_path}\n")
        f.write(f"  config   : {asdict(config)}\n")

    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Stage: normalize_2 (score-based denoise with centers)")
    ap.add_argument("--handwriting_id", required=True)

    ap.add_argument("--text_prob_thresh", type=float, default=0.25)
    ap.add_argument("--low_prob_thresh", type=float, default=0.02)
    ap.add_argument("--dist_thresh_px", type=float, default=40.0)
    ap.add_argument("--protect_dilate_r", type=int, default=8)
    ap.add_argument("--connectivity", type=int, default=8)
    ap.add_argument("--center_protect_radius_px", type=int, default=18)

    args = ap.parse_args()

    cfg = Normalize2Config(
        text_prob_thresh=args.text_prob_thresh,
        low_prob_thresh=args.low_prob_thresh,
        dist_thresh_px=args.dist_thresh_px,
        protect_dilate_r=args.protect_dilate_r,
        connectivity=args.connectivity,
        center_protect_radius_px=args.center_protect_radius_px,
    )

    run_normalize_2(handwriting_id=args.handwriting_id, config=cfg)
