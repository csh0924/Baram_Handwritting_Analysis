from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from Analysis_Module.common_paths import find_project_root, get_handwriting_dir
from Analysis_Module.io_unicode_cv import imread_unicode

def load_label_any(path: Union[str, Path]) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        a = np.load(str(path))
        if a.dtype != np.uint8:
            a = a.astype(np.uint8)
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        if a.ndim == 3 and a.shape[-1] == 1:
            a = a[..., 0]
        if a.ndim != 2:
            raise ValueError(f"Expected HxW label, got {a.shape} from {path}")
        return a

    m = imread_unicode(str(path), flags=cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    if m.dtype != np.uint8:
        m = m.astype(np.uint8)
    return m


def find_label_file(label_dir: Path, stem: str, priority=(".npy", ".png")) -> Optional[Path]:
    for ext in priority:
        p = label_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None



def binarize_label(label_0_3: np.ndarray, k: int) -> np.ndarray:
    return (label_0_3 == k).astype(np.uint8)

def area_score(aS: int, aP: int) -> float:
    if aS == 0 and aP == 0:
        return 1.0
    return float(min(aS, aP) / max(aS, aP))

def centroid_of_mask(mask01: np.ndarray) -> Optional[Tuple[float, float]]:
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))

def centroid_score(cS, cP, sigma: float = 15.0) -> float:
    if cS is None and cP is None:
        return 1.0
    if cS is None or cP is None:
        return 0.0
    dx = cS[0] - cP[0]
    dy = cS[1] - cP[1]
    d2 = dx * dx + dy * dy
    return float(np.exp(-d2 / (2.0 * sigma * sigma)))

def dilate_mask(mask01: np.ndarray, r: int = 3) -> np.ndarray:
    if r <= 0:
        return (mask01 > 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
    return cv2.dilate(mask01.astype(np.uint8), k, iterations=1)

def iou(maskA01: np.ndarray, maskB01: np.ndarray) -> float:
    A = (maskA01 > 0)
    B = (maskB01 > 0)
    inter = int(np.logical_and(A, B).sum())
    union = int(np.logical_or(A, B).sum())
    if union == 0:
        return 1.0
    return float(inter / union)

def dilated_iou(seed01: np.ndarray, pred01: np.ndarray, r: int = 3) -> float:
    Sd = dilate_mask(seed01, r=r)
    Pd = dilate_mask(pred01, r=r)
    return iou(Sd, Pd)

def bbox_of_labels(label_0_3: np.ndarray, ks=(1, 2, 3)) -> Optional[Tuple[int, int, int, int]]:
    m = np.zeros_like(label_0_3, dtype=np.uint8)
    for k in ks:
        m |= (label_0_3 == k).astype(np.uint8)
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return (x1, y1, x2, y2)


def paste_center(canvas_hw: Tuple[int, int], img: np.ndarray, center_xy: Tuple[float, float]) -> np.ndarray:
    H, W = canvas_hw
    out = np.zeros((H, W), dtype=img.dtype)

    h, w = img.shape[:2]
    cx, cy = center_xy

    x1 = int(round(cx - w / 2))
    y1 = int(round(cy - h / 2))
    x2 = x1 + w
    y2 = y1 + h

    sx1 = max(0, -x1)
    sy1 = max(0, -y1)
    sx2 = w - max(0, x2 - W)
    sy2 = h - max(0, y2 - H)

    dx1 = max(0, x1)
    dy1 = max(0, y1)
    dx2 = dx1 + (sx2 - sx1)
    dy2 = dy1 + (sy2 - sy1)

    if (sx2 > sx1) and (sy2 > sy1):
        out[dy1:dy2, dx1:dx2] = img[sy1:sy2, sx1:sx2]
    return out


def compute_isotropic_scale_from_bboxes(
    bbox_seed: Tuple[int, int, int, int],
    bbox_pred: Tuple[int, int, int, int],
    method: str = "geom_mean", 
) -> float:
    sx1, sy1, sx2, sy2 = bbox_seed
    px1, py1, px2, py2 = bbox_pred

    sw = max(1, sx2 - sx1)
    sh = max(1, sy2 - sy1)
    pw = max(1, px2 - px1)
    ph = max(1, py2 - py1)

    rw = pw / sw
    rh = ph / sh

    if method == "height":
        return float(rh)
    if method == "width":
        return float(rw)
    if method == "maxdim":
        return float(max(rw, rh))
    return float(np.sqrt(rw * rh))  


def scale_match_seed_to_pred(
    seed_label: np.ndarray,
    pred_label: np.ndarray,
    scale_method: str = "geom_mean",
    scale_clip: Tuple[float, float] = (0.85, 1.15),
) -> np.ndarray:
    H, W = pred_label.shape[:2]
    bboxS = bbox_of_labels(seed_label, ks=(1, 2, 3))
    bboxP = bbox_of_labels(pred_label, ks=(1, 2, 3))
    if (bboxS is None) or (bboxP is None):
        if seed_label.shape != (H, W):
            return cv2.resize(seed_label, (W, H), interpolation=cv2.INTER_NEAREST)
        return seed_label

    s = compute_isotropic_scale_from_bboxes(bboxS, bboxP, method=scale_method)
    s = float(np.clip(s, scale_clip[0], scale_clip[1]))

    new_w = max(1, int(round(seed_label.shape[1] * s)))
    new_h = max(1, int(round(seed_label.shape[0] * s)))
    seed_rs = cv2.resize(seed_label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    sx1, sy1, sx2, sy2 = bboxS
    px1, py1, px2, py2 = bboxP
    cP = ((px1 + px2) / 2.0, (py1 + py2) / 2.0)

    cS = (((sx1 + sx2) / 2.0) * s, ((sy1 + sy2) / 2.0) * s)

    bboxS_rs = bbox_of_labels(seed_rs, ks=(1, 2, 3))
    if bboxS_rs is None:
        return paste_center((H, W), seed_rs, center_xy=(W/2.0, H/2.0))
    rsx1, rsy1, rsx2, rsy2 = bboxS_rs
    cS_rs = ((rsx1 + rsx2) / 2.0, (rsy1 + rsy2) / 2.0)

    img_center = (seed_rs.shape[1] / 2.0, seed_rs.shape[0] / 2.0)
    delta = (cP[0] - cS_rs[0], cP[1] - cS_rs[1])
    paste_center_xy = (img_center[0] + delta[0], img_center[1] + delta[1])

    return paste_center((H, W), seed_rs, center_xy=paste_center_xy)


def rel_geometry_score(
    cS_a, cS_b, cP_a, cP_b,
    sigma_angle_deg: float = 20.0,
    sigma_len: float = 20.0,
) -> Optional[float]:
    if (cS_a is None) or (cS_b is None) or (cP_a is None) or (cP_b is None):
        return None

    vS = np.array([cS_b[0]-cS_a[0], cS_b[1]-cS_a[1]], dtype=np.float32)
    vP = np.array([cP_b[0]-cP_a[0], cP_b[1]-cP_a[1]], dtype=np.float32)

    lS = float(np.linalg.norm(vS))
    lP = float(np.linalg.norm(vP))
    if lS < 1e-6 or lP < 1e-6:
        return None

    dot = float(np.dot(vS, vP) / (lS * lP))
    dot = max(-1.0, min(1.0, dot))
    ang = float(np.degrees(np.arccos(dot)))  # 0..180
    score_angle = float(np.exp(-(ang * ang) / (2.0 * sigma_angle_deg * sigma_angle_deg)))

    dl = (lS - lP)
    score_len = float(np.exp(-(dl * dl) / (2.0 * sigma_len * sigma_len)))

    return 0.5 * score_angle + 0.5 * score_len


def evaluate_one_label(
    seed_label: np.ndarray,
    pred_label: np.ndarray,
    k: int,
    sigma_centroid: float = 15.0,
    dilate_r: int = 3,
) -> Dict:
    S = binarize_label(seed_label, k)
    P = binarize_label(pred_label, k)
    aS = int(S.sum())
    aP = int(P.sum())
    cS = centroid_of_mask(S)
    cP = centroid_of_mask(P)

    return {
        "area_seed": aS,
        "area_pred": aP,
        "area_score": area_score(aS, aP),
        "centroid_seed": cS,
        "centroid_pred": cP,
        "centroid_score": centroid_score(cS, cP, sigma=sigma_centroid),
        "iou_dilated": dilated_iou(S, P, r=dilate_r),
    }


def evaluate_pair(
    seed_label: np.ndarray,
    pred_label: np.ndarray,
    sigma_centroid: float = 15.0,
    dilate_r: int = 3,
    sigma_angle_deg: float = 20.0,
    sigma_len: float = 20.0,
) -> Dict[int, Dict]:
    out: Dict[int, Dict] = {}
    for k in (1, 2, 3):
        out[k] = evaluate_one_label(seed_label, pred_label, k, sigma_centroid, dilate_r)

    cS1, cS2, cS3 = out[1]["centroid_seed"], out[2]["centroid_seed"], out[3]["centroid_seed"]
    cP1, cP2, cP3 = out[1]["centroid_pred"], out[2]["centroid_pred"], out[3]["centroid_pred"]

    rel12 = rel_geometry_score(cS1, cS2, cP1, cP2, sigma_angle_deg=sigma_angle_deg, sigma_len=sigma_len)
    rel23 = rel_geometry_score(cS2, cS3, cP2, cP3, sigma_angle_deg=sigma_angle_deg, sigma_len=sigma_len)

    for k in (1, 2, 3):
        out[k]["rel_score_12"] = rel12
        out[k]["rel_score_23"] = rel23

    return out



@dataclass
class JamoEvalConfig:
    exts_priority: Tuple[str, ...] = (".npy", ".png")

    sigma_centroid: float = 15.0
    dilate_r: int = 3
    sigma_angle_deg: float = 20.0
    sigma_len: float = 20.0

    enable_scale_match: bool = True
    scale_method: str = "geom_mean"        
    scale_clip: Tuple[float, float] = (0.9, 1.1)  


@dataclass
class JamoEvalResult:
    handwriting_id: str
    printed_dir: str
    refined_dir: str
    out_txt: str
    compared: int
    skipped_missing: int
    config: dict


def evaluate_folders_mvp(
    printed_dir: Union[str, Path],
    refined_dir: Union[str, Path],
    config: Optional[JamoEvalConfig] = None,
    limit: Optional[int] = None,
) -> Tuple[List[Dict], int]:
    if config is None:
        config = JamoEvalConfig()

    printed_dir = Path(printed_dir)
    refined_dir = Path(refined_dir)

    stems: List[str] = []
    for p in printed_dir.iterdir():
        if p.is_file() and p.suffix.lower() in [e.lower() for e in config.exts_priority]:
            stems.append(p.stem)
    stems = sorted(set(stems))

    if limit is not None:
        stems = stems[: int(limit)]

    results: List[Dict] = []
    skipped = 0

    for stem in stems:
        p_seed = find_label_file(printed_dir, stem, config.exts_priority)
        p_pred = find_label_file(refined_dir, stem, config.exts_priority)
        if (p_seed is None) or (p_pred is None):
            skipped += 1
            continue

        seed = load_label_any(p_seed)
        pred = load_label_any(p_pred)
        
        if pred.shape != seed.shape:
            H, W = pred.shape[:2]
            seed = cv2.resize(seed, (W, H), interpolation=cv2.INTER_NEAREST)
        
        if config.enable_scale_match:
            seed = scale_match_seed_to_pred(
                seed_label=seed,
                pred_label=pred,
                scale_method=config.scale_method,
                scale_clip=config.scale_clip,
            )

        if pred.shape != seed.shape:
            H, W = seed.shape[:2]
            pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)

        scores = evaluate_pair(
            seed_label=seed,
            pred_label=pred,
            sigma_centroid=config.sigma_centroid,
            dilate_r=config.dilate_r,
            sigma_angle_deg=config.sigma_angle_deg,
            sigma_len=config.sigma_len,
        )

        results.append({
            "stem": stem,
            "seed_path": str(p_seed),
            "pred_path": str(p_pred),
            "scores": scores,
        })

    return results, skipped


def save_mvp_results_to_txt(results: List[Dict], out_txt_path: Union[str, Path]) -> None:
    out_txt_path = Path(out_txt_path)
    out_txt_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("# MVP comparison results (printed vs refined handwriting)")
    lines.append("# Per character, per label (1=초성, 2=중성, 3=종성)")
    lines.append("# Fields:")
    lines.append("#  area_score       : 면적 비율 일치도 (0~1)")
    lines.append("#  centroid_score   : 중심 위치 일치도 (0~1)")
    lines.append("#  iou_dilated      : 팽창 IoU (형태+위치, 0~1)")
    lines.append("#  rel_score_12     : 초-중 상대 구조 점수 (0~1)")
    lines.append("#  rel_score_23     : 중-종 상대 구조 점수 (0~1)")
    lines.append("")

    for r in results:
        stem = r["stem"]
        s = r["scores"]
        lines.append(f"[CHAR] {stem}")

        s1 = s[1]
        rel12 = "None" if s1["rel_score_12"] is None else f"{s1['rel_score_12']:.4f}"
        lines.append(
            f"  [초성] area_score={s1['area_score']:.4f}, centroid_score={s1['centroid_score']:.4f}, "
            f"iou_dilated={s1['iou_dilated']:.4f}, rel_score_12={rel12}"
        )

        s2 = s[2]
        rel12_2 = "None" if s2["rel_score_12"] is None else f"{s2['rel_score_12']:.4f}"
        rel23_2 = "None" if s2["rel_score_23"] is None else f"{s2['rel_score_23']:.4f}"
        lines.append(
            f"  [중성] area_score={s2['area_score']:.4f}, centroid_score={s2['centroid_score']:.4f}, "
            f"iou_dilated={s2['iou_dilated']:.4f}, rel_score_12={rel12_2}, rel_score_23={rel23_2}"
        )

        s3 = s[3]
        if s3["rel_score_23"] is not None:
            rel23_3 = f"{s3['rel_score_23']:.4f}"
            lines.append(
                f"  [종성] area_score={s3['area_score']:.4f}, centroid_score={s3['centroid_score']:.4f}, "
                f"iou_dilated={s3['iou_dilated']:.4f}, rel_score_23={rel23_3}"
            )

        lines.append("")

    out_txt_path.write_text("\n".join(lines), encoding="utf-8")


def run_jamo_eval(
    handwriting_id: str,
    config: Optional[JamoEvalConfig] = None,
    project_root: Optional[Union[str, Path]] = None,
    printed_dir: Optional[Union[str, Path]] = None,
    refined_dir: Optional[Union[str, Path]] = None,
    out_txt: Optional[Union[str, Path]] = None,
    limit: Optional[int] = None,
) -> JamoEvalResult:
    if config is None:
        config = JamoEvalConfig()

    project_root = Path(project_root) if project_root is not None else find_project_root(marker_dir="Analysis_Module")
    hw_dir = get_handwriting_dir(project_root, handwriting_id)

    if printed_dir is None:
        printed_dir = hw_dir / "printed_segments" / "labels_npy"
    else:
        printed_dir = Path(printed_dir)

    if refined_dir is None:
        refined_dir = hw_dir / "segments_refined" / "labels_npy"
    else:
        refined_dir = Path(refined_dir)

    if out_txt is None:
        out_txt = hw_dir / "jamo_eval" / "jamo_mvp.txt"
    else:
        out_txt = Path(out_txt)

    results, skipped = evaluate_folders_mvp(
        printed_dir=printed_dir,
        refined_dir=refined_dir,
        config=config,
        limit=limit,
    )
    save_mvp_results_to_txt(results, out_txt)

    return JamoEvalResult(
        handwriting_id=handwriting_id,
        printed_dir=str(printed_dir),
        refined_dir=str(refined_dir),
        out_txt=str(out_txt),
        compared=len(results),
        skipped_missing=skipped,
        config=asdict(config),
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Jamo evaluation: printed_segments vs segments_refined")
    ap.add_argument("--handwriting_id", required=True)
    ap.add_argument("--printed_dir", default=None)
    ap.add_argument("--refined_dir", default=None)
    ap.add_argument("--out_txt", default=None)
    ap.add_argument("--limit", type=int, default=None)

    ap.add_argument("--sigma_centroid", type=float, default=15.0)
    ap.add_argument("--dilate_r", type=int, default=3)
    ap.add_argument("--sigma_angle_deg", type=float, default=20.0)
    ap.add_argument("--sigma_len", type=float, default=20.0)

    args = ap.parse_args()

    cfg = JamoEvalConfig(
        sigma_centroid=args.sigma_centroid,
        dilate_r=args.dilate_r,
        sigma_angle_deg=args.sigma_angle_deg,
        sigma_len=args.sigma_len,
    )

    out = run_jamo_eval(
        handwriting_id=args.handwriting_id,
        config=cfg,
        printed_dir=args.printed_dir,
        refined_dir=args.refined_dir,
        out_txt=args.out_txt,
        limit=args.limit,
    )
    print(out)
