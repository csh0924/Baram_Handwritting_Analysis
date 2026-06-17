from __future__ import annotations

import time
import json
from dataclasses import dataclass, asdict
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from Analysis_Module.common_paths import find_project_root, get_handwriting_dir
from Analysis_Module.io_unicode_cv import imread_unicode

from Analysis_Module.craft_module import craft_utils, file_utils, imgproc
from Analysis_Module.craft_module.craft import CRAFT
from Analysis_Module.craft_module.refinenet import RefineNet
from Analysis_Module.errors import CenterDetectionError


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def load_craft_model(
    trained_model: str | Path,
    use_cuda: bool = True,
    use_refiner: bool = False,
    refiner_model: str | Path | None = None,
):
    if use_cuda and not torch.cuda.is_available():
        print("[WARN] CUDA is not available, fallback to CPU.")
        use_cuda = False

    net = CRAFT()

    trained_model = str(trained_model)
    print(f"Loading CRAFT weights: {trained_model}")

    if use_cuda:
        net.load_state_dict(copyStateDict(torch.load(trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(trained_model, map_location="cpu")))

    if use_cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    refine_net = None
    if use_refiner:
        if refiner_model is None:
            raise ValueError("use_refiner=True이면 refiner_model 경로를 지정해야 합니다.")
        refine_net = RefineNet()
        refiner_model = str(refiner_model)
        print(f"Loading Refiner weights: {refiner_model}")

        if use_cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(refiner_model, map_location="cpu")))

        refine_net.eval()

    return net, refine_net, use_cuda


def extract_centers_from_score_text(score_text: np.ndarray, thr: float = 0.7, min_area: int = 3):
    _, binary = cv2.threshold(score_text.astype(np.float32), thr, 1.0, cv2.THRESH_BINARY)
    binary = (binary * 255).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    centers = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        cx, cy = centroids[i]
        centers.append((cx, cy))
    return centers


def merge_centers_by_gap_to_expected(centers, expected_count: int):
    if expected_count <= 0:
        return centers

    if len(centers) <= expected_count:
        return sorted(centers, key=lambda p: p[0])

    centers_sorted = sorted(centers, key=lambda p: p[0])
    n = len(centers_sorted)

    dx = [centers_sorted[i + 1][0] - centers_sorted[i][0] for i in range(n - 1)]
    k = expected_count - 1
    if k <= 0:
        mx = sum(p[0] for p in centers_sorted) / n
        my = sum(p[1] for p in centers_sorted) / n
        return [(mx, my)]

    idx_sorted = sorted(range(len(dx)), key=lambda i: dx[i], reverse=True)
    boundaries = sorted(idx_sorted[:k])

    groups = []
    start = 0
    for b in boundaries:
        groups.append(centers_sorted[start : b + 1])
        start = b + 1
    groups.append(centers_sorted[start:])

    merged = []
    for g in groups:
        mx = sum(p[0] for p in g) / len(g)
        my = sum(p[1] for p in g) / len(g)
        merged.append((mx, my))
    return merged


def detect_text_and_centers(
    net,
    image,
    *,
    text_threshold=0.5,
    link_threshold=0.1,
    low_text=0.2,
    use_cuda=True,
    poly=False,
    refine_net=None,
    canvas_size=1280,
    mag_ratio=1.5,
    show_time=False,
    center_thr=0.4,
    center_min_area=30,
    expected_char_count: int,
):
    t0 = time.time()

    if isinstance(image, Image.Image):
        image = np.array(image)

    H0, W0 = image.shape[:2]

    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio
    )
    ratio_h = ratio_w = 1 / target_ratio

    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    if use_cuda:
        x = x.cuda()

    with torch.no_grad():
        y, feature = net(x)

    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

    infer_time = time.time() - t0

    pad_h, pad_w = img_resized.shape[:2]
    valid_w = int(round(W0 * target_ratio))
    valid_h = int(round(H0 * target_ratio))

    heat_h, heat_w = score_text.shape[:2]
    valid_heat_w = int(round(heat_w * (valid_w / pad_w)))
    valid_heat_h = int(round(heat_h * (valid_h / pad_h)))
    valid_heat_w = max(1, min(valid_heat_w, heat_w))
    valid_heat_h = max(1, min(valid_heat_h, heat_h))

    score_text_valid = score_text[:valid_heat_h, :valid_heat_w]

    score_text_orig = cv2.resize(score_text_valid, (W0, H0), interpolation=cv2.INTER_LINEAR)

    centers_raw = extract_centers_from_score_text(score_text_orig, thr=center_thr, min_area=center_min_area)

    if len(centers_raw) < expected_char_count:
        raise CenterDetectionError(
            "Too few centers detected",
            expected=expected_char_count,
            detected=len(centers_raw),
            raw_detected=len(centers_raw),
        )

    if len(centers_raw) > expected_char_count:
        centers = merge_centers_by_gap_to_expected(centers_raw, expected_char_count)
    else:
        centers = sorted(centers_raw, key=lambda p: p[0])

    if len(centers) != expected_char_count:
        raise CenterDetectionError(
            "Center normalization failed",
            expected=expected_char_count,
            detected=len(centers),
            raw_detected=len(centers_raw),
        )

    if show_time:
        print(f"infer time: {infer_time:.3f}s")

    return centers, score_text_orig


@dataclass
class CenterConfig:
    text_threshold: float = 0.5
    link_threshold: float = 0.1
    low_text: float = 0.2
    canvas_size: int = 1280
    mag_ratio: float = 1.5
    poly: bool = False
    show_time: bool = True
    center_thr: float = 0.4
    center_min_area: int = 30
    use_cuda: bool = True
    use_refiner: bool = False


def _expected_from_text(original_text: str) -> Tuple[str, int]:
    cleaned = original_text.replace(" ", "")
    return cleaned, len(cleaned)


def run_center(
    handwriting_id: str,
    original_text: str,
    normalized_image_path: Optional[str | Path] = None,
    config: Optional[CenterConfig] = None,
    project_root: Optional[Path] = None,
):
    config = config or CenterConfig()
    project_root = project_root or find_project_root(marker_dir="Analysis_Module")

    handwriting_dir = get_handwriting_dir(project_root, handwriting_id)
    handwriting_dir.mkdir(parents=True, exist_ok=True)

    if normalized_image_path is None:
        normalized_image_path = handwriting_dir / "normalize_1" / "normalized.png"
    else:
        normalized_image_path = Path(normalized_image_path)

    if not normalized_image_path.exists():
        raise FileNotFoundError(f"normalized 이미지가 없습니다: {normalized_image_path}")

    center_dir = handwriting_dir / "center"
    score_dir = handwriting_dir / "score"
    center_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)

    out_center_txt = center_dir / "center.txt"
    out_score_npy = score_dir / "score.npy"
    manifest_path = center_dir / "manifest.json"

    cleaned_text, expected_count = _expected_from_text(original_text)
    if expected_count <= 0:
        raise ValueError("original_text에서 유효한 문자 수를 계산할 수 없습니다(길이 0).")

    craft_weights = project_root / "Analysis_Module" / "craft_module" / "weights" / "craft_mlt_25k.pth"
    refiner_weights = project_root / "Analysis_Module" / "craft_module" / "weights" / "craft_refiner_CTW1500.pth"

    net, refine_net, use_cuda = load_craft_model(
        trained_model=craft_weights,
        use_cuda=config.use_cuda,
        use_refiner=config.use_refiner,
        refiner_model=refiner_weights if config.use_refiner else None,
    )

    image = imgproc.loadImage(str(normalized_image_path))
    centers, score_text_orig = detect_text_and_centers(
        net=net,
        image=image,
        text_threshold=config.text_threshold,
        link_threshold=config.link_threshold,
        low_text=config.low_text,
        use_cuda=use_cuda,
        poly=config.poly,
        refine_net=refine_net,
        canvas_size=config.canvas_size,
        mag_ratio=config.mag_ratio,
        show_time=config.show_time,
        center_thr=config.center_thr,
        center_min_area=config.center_min_area,
        expected_char_count=expected_count,
    )

    np.save(out_score_npy, score_text_orig.astype(np.float32))

    centers_sorted = sorted(centers, key=lambda p: p[0])
    with open(out_center_txt, "w", encoding="utf-8") as f:
        for ch, (cx, cy) in zip(cleaned_text, centers_sorted):
            f.write(f"{ch}: {cx:.2f},{cy:.2f}\n")

    manifest = {
        "handwriting_id": handwriting_id,
        "normalized_image_path": str(normalized_image_path),
        "original_text": original_text,
        "out_center_txt": str(out_center_txt),
        "out_score_npy": str(out_score_npy),
        "config": asdict(config),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest
