from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union, Any

import cv2
import numpy as np

from Analysis_Module.common_paths import find_project_root, get_handwriting_dir
from Analysis_Module.io_unicode_cv import imread_unicode, imwrite_unicode


def _list_images(folder: Path, exts: Tuple[str, ...]) -> List[Path]:
    folder = Path(folder)
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: p.name)
    return files


def load_label_any(path: Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(str(path))
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return arr
    img = imread_unicode(str(path), flags=cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.uint8)


def _find_label_file(label_dir: Path, stem: str, priority: Tuple[str, ...]) -> Optional[Path]:
    for ext in priority:
        p = label_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _ensure_label_dirs(root: Path) -> None:
    (root / "labels_png").mkdir(parents=True, exist_ok=True)
    (root / "labels_npy").mkdir(parents=True, exist_ok=True)


def bbox_from_label(label_0_3: np.ndarray, k: int) -> Optional[Dict[str, int]]:
    ys, xs = np.where(label_0_3 == k)
    if xs.size == 0 or ys.size == 0:
        return None
    x1 = int(xs.min())
    x2 = int(xs.max())
    y1 = int(ys.min())
    y2 = int(ys.max())
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def make_ink_mask_white_on_black(
    img_rgb: np.ndarray,
    thr: int = 160,
    blur_ksize: int = 0,
    morph_open: int = 0,
    morph_close: int = 0,
) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    if blur_ksize and blur_ksize > 1:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    ink = (gray > thr).astype(np.uint8) * 255

    if morph_open and morph_open > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))
        ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, k)

    if morph_close and morph_close > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, k)

    return ink


def refine_with_seed_distance(
    ink_mask_u8: np.ndarray,
    pred_label: np.ndarray,
    seed_label: np.ndarray,
    *,
    max_dist: float = 40.0,
    allow_classes: Tuple[int, ...] = (1, 2, 3),
    fallback_to_pred: bool = True,
) -> np.ndarray:
    ink = (ink_mask_u8 > 0)

    candidates: List[int] = []
    dist_maps: List[np.ndarray] = []

    for k in allow_classes:
        if np.any(seed_label == k):
            candidates.append(k)
            seed_k = (seed_label == k).astype(np.uint8)
            inv = (1 - seed_k) * 255
            dist = cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3)
            dist_maps.append(dist)

    if len(candidates) == 0:
        return pred_label.copy()

    stack = np.stack(dist_maps, axis=0)
    min_dist = np.min(stack, axis=0)
    arg = np.argmin(stack, axis=0)
    assigned = np.take(np.array(candidates, dtype=np.uint8), arg)

    refined = pred_label.copy()
    refined[ink] = assigned[ink]

    far = ink & (min_dist > max_dist)
    if fallback_to_pred:
        refined[far] = pred_label[far]
    else:
        refined[far] = 0

    return refined


def cleanup_small_label_fragments_in_components(
    ink_mask_u8: np.ndarray,
    label_0_3: np.ndarray,
    *,
    min_ratio: float = 0.03,
    min_pixels: int = 15,
    connectivity: int = 8,
) -> np.ndarray:
    ink = (ink_mask_u8 > 0).astype(np.uint8)
    num, cc = cv2.connectedComponents(ink, connectivity=connectivity)
    out = label_0_3.copy()

    for cid in range(1, num):
        region = (cc == cid)
        if not np.any(region):
            continue

        vals, counts = np.unique(out[region], return_counts=True)
        total = int(region.sum())

        best_label = 0
        best_count = -1
        for v, c in zip(vals, counts):
            if int(v) == 0:
                continue
            if int(c) > best_count:
                best_count = int(c)
                best_label = int(v)

        if best_label == 0:
            continue

        for v, c in zip(vals, counts):
            v = int(v)
            c = int(c)
            if v == 0 or v == best_label:
                continue
            if c < min_pixels or (c / max(total, 1)) < min_ratio:
                out[region & (out == v)] = best_label

    return out


def suppress_minor_labels_in_components(
    ink_mask_u8: np.ndarray,
    label_0_3: np.ndarray,
    *,
    protect_classes: Tuple[int, ...] = (1, 2, 3),
    min_ratio: float = 0.06,
    min_pixels: int = 20,
    connectivity: int = 8,
) -> np.ndarray:
    assert ink_mask_u8.shape == label_0_3.shape
    ink = (ink_mask_u8 > 0)

    out = label_0_3.copy().astype(np.uint8)
    out[~ink] = 0

    conn = 8 if connectivity == 8 else 4
    num, cc = cv2.connectedComponents(ink.astype(np.uint8), connectivity=conn)

    for cid in range(1, num):
        m = (cc == cid)

        total_allowed = 0
        counts: Dict[int, int] = {}
        for k in protect_classes:
            c = int((out[m] == k).sum())
            counts[k] = c
            total_allowed += c

        if total_allowed == 0:
            continue

        majority = max(counts.keys(), key=lambda k: counts[k])

        for k in protect_classes:
            if k == majority:
                continue
            c = counts[k]
            if c == 0:
                continue
            ratio = c / float(total_allowed)
            if (c < min_pixels) or (ratio < min_ratio):
                out[m & (out == k)] = majority

    out[~ink] = 0
    return out


@dataclass
class RefineSegmentsConfig:
    file_exts_img: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    pred_ext_priority: Tuple[str, ...] = (".npy", ".png")
    seed_ext_priority: Tuple[str, ...] = (".npy", ".png")

    ink_thr: int = 160
    ink_blur: int = 0
    ink_open: int = 0
    ink_close: int = 0

    max_dist: float = 8.0
    fallback_to_pred: bool = True

    do_cleanup: bool = True
    cleanup_min_ratio: float = 0.03
    cleanup_min_pixels: int = 15

    suppress_min_ratio: float = 0.30
    suppress_min_pixels: int = 20
    suppress_connectivity: int = 8

    preserve_disappeared_labels: bool = True
    protect_classes: Tuple[int, ...] = (1, 2, 3)

    out_subdir_name: str = "segments_refined"
    save_masked_vis: bool = False

    save_bboxes: bool = True
    bboxes_subdir: str = "bboxes"
    bboxes_filename: str = "jamo_bboxes.json"


@dataclass
class RefineSegmentsResult:
    handwriting_id: str
    hw_img_dir: str
    hw_pred_dir: str
    seed_dir: str
    out_dir: str
    processed: int
    skipped_missing: int
    config: dict
    manifest_path: str
    bboxes_path: str


def _save_masked_vis(out_path: Path, label_0_3: np.ndarray, ink_mask_u8: np.ndarray) -> None:
    palette = np.array(
        [[0, 0, 0],
         [255, 0, 0],
         [0, 255, 0],
         [0, 0, 255]], dtype=np.uint8
    )
    m = np.clip(label_0_3.astype(np.int32), 0, 3)
    colored = palette[m]
    out = np.zeros_like(colored)
    ink = (ink_mask_u8 > 0)
    out[ink] = colored[ink]
    imwrite_unicode(out_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


def run_refine_segments(
    handwriting_id: str,
    config: Optional[RefineSegmentsConfig] = None,
    project_root: Optional[Union[str, Path]] = None,
    *,
    hw_img_dir: Optional[Union[str, Path]] = None,
    hw_pred_dir: Optional[Union[str, Path]] = None,
    seed_dir: Optional[Union[str, Path]] = None,
    out_dir: Optional[Union[str, Path]] = None,
) -> RefineSegmentsResult:
    if config is None:
        config = RefineSegmentsConfig()

    if project_root is None:
        project_root = find_project_root(marker_dir="Analysis_Module")
    else:
        project_root = Path(project_root)

    handwriting_dir = get_handwriting_dir(project_root, handwriting_id)

    if hw_img_dir is None:
        hw_img_dir = handwriting_dir / "char_cropped"
    else:
        hw_img_dir = Path(hw_img_dir)

    if hw_pred_dir is None:
        hw_pred_dir = handwriting_dir / "segments" / "labels_npy"
    else:
        hw_pred_dir = Path(hw_pred_dir)

    if seed_dir is None:
        seed_dir = handwriting_dir / "printed_segments" / "labels_npy"
    else:
        seed_dir = Path(seed_dir)

    if out_dir is None:
        out_dir = handwriting_dir / config.out_subdir_name
    else:
        out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_label_dirs(out_dir)

    if config.save_masked_vis:
        (out_dir / "masked").mkdir(parents=True, exist_ok=True)

    jamo_bbox_root = handwriting_dir / "jamo_bboxes"
    if config.save_bboxes:
        jamo_bbox_root.mkdir(parents=True, exist_ok=True)

    all_bboxes: Dict[str, Any] = {
        "handwriting_id": handwriting_id,
        "jamo_bbox_root": str(jamo_bbox_root),
        "items": [],
    }

    img_files = _list_images(Path(hw_img_dir), config.file_exts_img)
    if len(img_files) == 0:
        raise RuntimeError(f"No images in: {hw_img_dir}")

    processed = 0
    skipped_missing = 0

    for img_path in img_files:
        stem = img_path.stem

        pred_path = _find_label_file(Path(hw_pred_dir), stem, config.pred_ext_priority)
        seed_path = _find_label_file(Path(seed_dir), stem, config.seed_ext_priority)

        if pred_path is None or seed_path is None:
            skipped_missing += 1
            continue

        bgr = imread_unicode(str(img_path), flags=cv2.IMREAD_COLOR)
        if bgr is None:
            skipped_missing += 1
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        pred = load_label_any(pred_path)
        seed = load_label_any(seed_path)

        H, W = rgb.shape[:2]
        if pred.shape != (H, W):
            pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)
        if seed.shape != (H, W):
            seed = cv2.resize(seed, (W, H), interpolation=cv2.INTER_NEAREST)

        ink = make_ink_mask_white_on_black(
            rgb,
            thr=config.ink_thr,
            blur_ksize=config.ink_blur,
            morph_open=config.ink_open,
            morph_close=config.ink_close,
        )
        ink_bool = (ink > 0)

        clustered = refine_with_seed_distance(
            ink_mask_u8=ink,
            pred_label=pred,
            seed_label=seed,
            max_dist=config.max_dist,
            allow_classes=config.protect_classes,
            fallback_to_pred=config.fallback_to_pred,
        )
        clustered[~ink_bool] = 0

        if config.do_cleanup:
            clustered = cleanup_small_label_fragments_in_components(
                ink_mask_u8=ink,
                label_0_3=clustered,
                min_ratio=config.cleanup_min_ratio,
                min_pixels=config.cleanup_min_pixels,
                connectivity=8,
            )
            clustered[~ink_bool] = 0

        before_labels = set(int(v) for v in np.unique(clustered[ink_bool]) if int(v) in config.protect_classes)

        suppressed = suppress_minor_labels_in_components(
            ink_mask_u8=ink,
            label_0_3=clustered,
            protect_classes=config.protect_classes,
            min_ratio=config.suppress_min_ratio,
            min_pixels=config.suppress_min_pixels,
            connectivity=config.suppress_connectivity,
        )
        suppressed[~ink_bool] = 0

        after_labels = set(int(v) for v in np.unique(suppressed[ink_bool]) if int(v) in config.protect_classes)
        disappeared = sorted(list(before_labels - after_labels))

        if config.preserve_disappeared_labels and disappeared:
            for k in disappeared:
                restore_mask = ink_bool & (clustered == k)
                if np.any(restore_mask):
                    suppressed[restore_mask] = k
            suppressed[~ink_bool] = 0

        bboxes_item = {
            "stem": stem,
            "path_img": str(img_path),
            "bboxes": {
                "chosung": bbox_from_label(suppressed, 1),
                "jungsung": bbox_from_label(suppressed, 2),
                "jongsung": bbox_from_label(suppressed, 3),
            },
        }

        if config.save_bboxes:
            per_path = jamo_bbox_root / f"{stem}.json"
            with open(per_path, "w", encoding="utf-8") as f:
                json.dump(bboxes_item, f, ensure_ascii=False, indent=2)

        all_bboxes["items"].append(bboxes_item)

        imwrite_unicode(out_dir / "labels_png" / f"{stem}.png", suppressed.astype(np.uint8))
        np.save(str(out_dir / "labels_npy" / f"{stem}.npy"), suppressed.astype(np.uint8))

        if config.save_masked_vis:
            _save_masked_vis(out_dir / "masked" / f"{stem}.png", suppressed, ink)

        processed += 1

    bboxes_path = out_dir / config.bboxes_filename
    if config.save_bboxes:
        with open(bboxes_path, "w", encoding="utf-8") as f:
            json.dump(all_bboxes, f, ensure_ascii=False, indent=2)

    manifest_path = out_dir / "refine_manifest.json"
    result = RefineSegmentsResult(
        handwriting_id=handwriting_id,
        hw_img_dir=str(hw_img_dir),
        hw_pred_dir=str(hw_pred_dir),
        seed_dir=str(seed_dir),
        out_dir=str(out_dir),
        processed=processed,
        skipped_missing=skipped_missing,
        config=asdict(config),
        manifest_path=str(manifest_path),
        bboxes_path=str(bboxes_path),
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)

    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser("refine_segments: save final to segments_refined (no overwrite)")
    ap.add_argument("--handwriting_id", required=True)

    ap.add_argument("--ink_thr", type=int, default=160)
    ap.add_argument("--max_dist", type=float, default=8.0)
    ap.add_argument("--suppress_min_ratio", type=float, default=0.30)

    ap.add_argument("--save_masked", action="store_true")
    ap.add_argument("--out_subdir", default="segments_refined")

    ap.add_argument("--no_bboxes", action="store_true")

    args = ap.parse_args()

    cfg = RefineSegmentsConfig(
        ink_thr=args.ink_thr,
        max_dist=args.max_dist,
        suppress_min_ratio=args.suppress_min_ratio,
        save_masked_vis=args.save_masked,
        out_subdir_name=args.out_subdir,
        save_bboxes=(not args.no_bboxes),
    )
    out = run_refine_segments(handwriting_id=args.handwriting_id, config=cfg)
    print(out)
