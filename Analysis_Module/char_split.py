from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from collections import deque

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
                raise ValueError(f"Invalid line (missing ':') at {line_no}: {line}")
            ch, rest = line.split(":", 1)
            ch = ch.strip()
            rest = rest.strip()
            if "," not in rest:
                raise ValueError(f"Invalid line (missing ',') at {line_no}: {line}")
            xs, ys = rest.split(",", 1)
            x = float(xs.strip())
            y = float(ys.strip())
            out.append((ch, x, y))

    if not out:
        raise ValueError(f"center.txt is empty: {center_txt}")

    return out


def voronoi_label_map(H: int, W: int, centers_xy: np.ndarray, connectivity: int = 4) -> np.ndarray:
    if connectivity == 4:
        neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neigh = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    label = np.zeros((H, W), np.int32)
    dist = np.full((H, W), 1_000_000_000, np.int32)
    q = deque()

    for i, (x, y) in enumerate(centers_xy, start=1):
        sx, sy = int(round(x)), int(round(y))
        sx = int(np.clip(sx, 0, W - 1))
        sy = int(np.clip(sy, 0, H - 1))
        label[sy, sx] = i
        dist[sy, sx] = 0
        q.append((sx, sy))

    while q:
        x, y = q.popleft()
        for dx, dy in neigh:
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= W or ny >= H:
                continue
            if dist[ny, nx] > dist[y, x] + 1:
                dist[ny, nx] = dist[y, x] + 1
                label[ny, nx] = label[y, x]
                q.append((nx, ny))

    return label


def ink_mask_auto(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    _, m_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, m_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    ink_bin = (m_bin > 0).astype(np.uint8)
    ink_inv = (m_inv > 0).astype(np.uint8)

    return ink_bin if ink_bin.sum() <= ink_inv.sum() else ink_inv


def assign_split_ink_components(
    label_map: np.ndarray,
    img: np.ndarray,
    centers_xy: np.ndarray,
    *,
    connectivity: int = 8,
    min_comp_area: int = 8,
    w_area: float = 1.0,
    w_dist: float = 2.0,
    majority_ratio_thresh: float = 0.8,
) -> np.ndarray:
    refined = label_map.copy()
    centers_xy = np.asarray(centers_xy, dtype=np.float32)

    ink01 = ink_mask_auto(img).astype(np.uint8)
    conn = 4 if connectivity == 4 else 8

    num, cc = cv2.connectedComponents(ink01, connectivity=conn)
    if num <= 1:
        return refined

    for cid in range(1, num):
        comp_mask = (cc == cid)
        area = int(comp_mask.sum())
        if area < min_comp_area:
            continue

        labs = refined[comp_mask]
        labs = labs[labs != 0]
        if labs.size == 0:
            continue

        uniq, cnts = np.unique(labs, return_counts=True)
        if uniq.size <= 1:
            continue

        total = float(cnts.sum())
        major_idx = int(np.argmax(cnts))
        major_lab = int(uniq[major_idx])
        major_ratio = float(cnts[major_idx]) / total

        if majority_ratio_thresh is not None and major_ratio >= float(majority_ratio_thresh):
            refined[comp_mask] = major_lab
            continue

        ys, xs = np.where(comp_mask)
        mx, my = float(xs.mean()), float(ys.mean())

        best_lab = None
        best_score = None
        for lab, lab_cnt in zip(uniq.tolist(), cnts.tolist()):
            cx, cy = float(centers_xy[lab - 1][0]), float(centers_xy[lab - 1][1])
            d2 = (mx - cx) ** 2 + (my - cy) ** 2
            score = (w_area * float(lab_cnt)) - (w_dist * float(d2))
            if best_score is None or score > best_score:
                best_score = score
                best_lab = int(lab)

        refined[comp_mask] = best_lab if best_lab is not None else major_lab

    return refined


def assign_global_ink_cc_to_majority_label(
    label_map: np.ndarray,
    img: np.ndarray,
    *,
    connectivity: int = 8,
    min_comp_area: int = 8,
    max_intrusion_pixels: int | None = 30,
    max_intrusion_ratio: float | None = None,
) -> np.ndarray:
    refined = label_map.copy()
    ink01 = ink_mask_auto(img).astype(np.uint8)
    conn = 4 if connectivity == 4 else 8

    num, cc = cv2.connectedComponents(ink01, connectivity=conn)
    if num <= 1:
        return refined

    for cid in range(1, num):
        comp = (cc == cid)
        area = int(comp.sum())
        if area < min_comp_area:
            continue

        labs = refined[comp]
        labs = labs[labs != 0]
        if labs.size == 0:
            continue

        uniq, cnts = np.unique(labs, return_counts=True)
        if uniq.size <= 1:
            continue

        total = int(cnts.sum())
        major_idx = int(np.argmax(cnts))
        major_lab = int(uniq[major_idx])
        major_cnt = int(cnts[major_idx])
        intrusion = total - major_cnt

        if max_intrusion_pixels is not None and intrusion > int(max_intrusion_pixels):
            continue
        if max_intrusion_ratio is not None and (intrusion / float(total)) > float(max_intrusion_ratio):
            continue

        refined[comp] = major_lab

    return refined


def save_regions_as_images_and_build_roi_map(
    *,
    img: np.ndarray,
    label_map: np.ndarray,
    out_dir: Path,
    filename_stem: str,
    pad: int = 4,
    min_pixels: int = 80,
    save_fullsize: bool = False,
) -> Dict[str, Tuple[int, int, int, int]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    H, W = label_map.shape[:2]
    labels = [int(l) for l in np.unique(label_map) if l != 0]

    roi_map: Dict[str, Tuple[int, int, int, int]] = {}

    for lab in labels:
        mask = (label_map == lab)
        pix_count = int(mask.sum())
        if pix_count < min_pixels:
            continue

        if save_fullsize:
            out = np.zeros_like(img)
            out[mask] = img[mask]
            roi_xyxy = (0, 0, W - 1, H - 1)
        else:
            ys, xs = np.where(mask)
            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(xs.min()), int(xs.max())

            y0 = max(0, y0 - pad)
            y1 = min(H - 1, y1 + pad)
            x0 = max(0, x0 - pad)
            x1 = min(W - 1, x1 + pad)

            roi = img[y0:y1 + 1, x0:x1 + 1]
            roi_mask = mask[y0:y1 + 1, x0:x1 + 1]

            out = np.zeros_like(roi)
            out[roi_mask] = roi[roi_mask]
            roi_xyxy = (x0, y0, x1, y1)

        out_path = out_dir / f"{filename_stem}_L{lab:03d}.png"
        imwrite_unicode(out_path, out)

        roi_map[out_path.name] = tuple(map(int, roi_xyxy))

    return roi_map


@dataclass
class CharSplitConfig:
    voronoi_connectivity: int = 4
    cc_connectivity: int = 8
    min_comp_area: int = 8
    w_area: float = 1.0
    w_dist: float = 2.0
    majority_ratio_thresh: float = 0.8

    global_cc_min_area: int = 8
    max_intrusion_pixels: int | None = 30
    max_intrusion_ratio: float | None = None

    region_pad: int = 4
    region_min_pixels: int = 80
    region_save_fullsize: bool = False

    save_roi_map_to_manifest: bool = False


@dataclass
class CharSplitResult:
    handwriting_id: str
    in_image_path: str
    center_txt: str
    out_dir: str
    manifest_path: str
    saved_count: int
    roi_map: Dict[str, Tuple[int, int, int, int]]
    config: dict


def run_char_split(
    handwriting_id: str,
    config: Optional[CharSplitConfig] = None,
    project_root: Optional[Path] = None,
    in_image_path: Optional[str | Path] = None,
    center_txt_path: Optional[str | Path] = None,
) -> CharSplitResult:
    if config is None:
        config = CharSplitConfig()

    if project_root is None:
        project_root = find_project_root(marker_dir="Analysis_Module")
    else:
        project_root = Path(project_root)

    handwriting_dir = get_handwriting_dir(project_root, handwriting_id)

    if in_image_path is None:
        in_image_path = handwriting_dir / "normalize_2" / "normalized.png"
    else:
        in_image_path = Path(in_image_path)

    if center_txt_path is None:
        center_txt_path = handwriting_dir / "center" / "center.txt"
    else:
        center_txt_path = Path(center_txt_path)

    if not Path(in_image_path).exists():
        raise FileNotFoundError(f"normalize_2 image not found: {in_image_path}")
    if not Path(center_txt_path).exists():
        raise FileNotFoundError(f"center.txt not found: {center_txt_path}")

    out_dir = handwriting_dir / "chars"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    log_path = out_dir / "log.txt"

    img = imread_unicode(str(in_image_path), flags=cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {in_image_path}")

    centers = read_centers_char_format(center_txt_path)
    centers_xy = np.array([[x, y] for (_, x, y) in centers], dtype=np.float32)
    if centers_xy.size == 0:
        raise ValueError("Empty centers.")

    H, W = img.shape[:2]

    label_voro = voronoi_label_map(H, W, centers_xy, connectivity=config.voronoi_connectivity)

    label_owner = assign_split_ink_components(
        label_voro, img, centers_xy,
        connectivity=config.cc_connectivity,
        min_comp_area=config.min_comp_area,
        w_area=config.w_area,
        w_dist=config.w_dist,
        majority_ratio_thresh=config.majority_ratio_thresh,
    )

    label_global = assign_global_ink_cc_to_majority_label(
        label_owner, img,
        connectivity=config.cc_connectivity,
        min_comp_area=config.global_cc_min_area,
        max_intrusion_pixels=config.max_intrusion_pixels,
        max_intrusion_ratio=config.max_intrusion_ratio,
    )

    label_final = assign_split_ink_components(
        label_global, img, centers_xy,
        connectivity=config.cc_connectivity,
        min_comp_area=config.min_comp_area,
        w_area=config.w_area,
        w_dist=config.w_dist,
        majority_ratio_thresh=config.majority_ratio_thresh,
    )

    roi_map = save_regions_as_images_and_build_roi_map(
        img=img,
        label_map=label_final,
        out_dir=out_dir,
        filename_stem="char",
        pad=config.region_pad,
        min_pixels=config.region_min_pixels,
        save_fullsize=config.region_save_fullsize,
    )

    saved_count = len(roi_map)

    result = CharSplitResult(
        handwriting_id=handwriting_id,
        in_image_path=str(in_image_path),
        center_txt=str(center_txt_path),
        out_dir=str(out_dir),
        manifest_path=str(manifest_path),
        saved_count=saved_count,
        roi_map=roi_map,
        config=asdict(config),
    )

    payload = asdict(result)
    if not config.save_roi_map_to_manifest:
        payload.pop("roi_map", None)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("[OK] char_split\n")
        f.write(f"  in_image: {in_image_path}\n")
        f.write(f"  centers : {center_txt_path}\n")
        f.write(f"  out_dir : {out_dir}\n")
        f.write(f"  saved   : {saved_count}\n")

    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Stage: char_split (Voronoi + CC reclaim) (roi_map in-memory)")
    ap.add_argument("--handwriting_id", required=True)
    args = ap.parse_args()

    run_char_split(handwriting_id=args.handwriting_id)