from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import re
import statistics


@dataclass(frozen=True)
class SizeScoreConfig:
    jong_height_ratio: float = 0.70

    tol_ratio_x: float = 0.12   
    tol_ratio_y: float = 0.12 

    zero_ratio_x: float = 0.45
    zero_ratio_y: float = 0.45

    power: float = 2.0

    w_x: float = 0.5
    w_y: float = 0.5

    base_method: str = "median"

    bbox_re: str = r"xyxy=\((\d+),(\d+),(\d+),(\d+)\)"


def has_jongseong(ch: str) -> Optional[bool]:
    if len(ch) != 1:
        return None
    code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:
        jong = (code - 0xAC00) % 28
        return jong != 0
    return None


def _robust_base(vals: List[float], method: str) -> Optional[float]:
    if not vals:
        return None
    if method == "mean":
        return float(sum(vals) / len(vals))
    return float(statistics.median(vals))


def _score_from_rel_error(rel_err: float, slot_part: float, tol: float, zero: float, power: float) -> Tuple[float, float]:
    if zero <= tol:
        raise ValueError("zero must be greater than tol.")

    if rel_err <= tol:
        return slot_part, 0.0
    if rel_err >= zero:
        return 0.0, 1.0

    z = (rel_err - tol) / (zero - tol)
    s = slot_part * (1.0 - (z ** power))
    if s < 0.0:
        s = 0.0
    elif s > slot_part:
        s = slot_part
    return s, z


def parse_charbbox_text(text: str, cfg: SizeScoreConfig = SizeScoreConfig()) -> List[Dict]:
    bbox_pat = re.compile(cfg.bbox_re)
    file_pat = re.compile(r"file\s*=\s*(.+)$")

    lines = text.splitlines()

    out: List[Dict] = []
    cur_file: Optional[str] = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        m_file = file_pat.match(line)
        if m_file:
            cur_file = m_file.group(1).strip()
            continue

        m_bbox = bbox_pat.search(line)
        if m_bbox:
            x1, y1, x2, y2 = map(int, m_bbox.groups())
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            out.append({
                "file": cur_file,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "w": w, "h": h,
            })

    return out


def compute_size_score_distributed(
    char_boxes: List[Dict],
    original_text: str,
    cfg: SizeScoreConfig = SizeScoreConfig(),
) -> Dict:
    n = len(char_boxes)
    if n <= 0:
        return {
            "score": 0.0,
            "count": 0,
            "slot": 0.0,
            "details": [],
            "config": cfg.__dict__,
            "warning": "No char bboxes found.",
        }

    chars = [c for c in original_text if not c.isspace()]
    if len(chars) < n:
        chars = chars + ["?"] * (n - len(chars))
    elif len(chars) > n:
        chars = chars[:n]

    widths = [cb["w"] for cb in char_boxes]
    base_w = _robust_base(widths, cfg.base_method)

    heights_nojong: List[float] = []
    heights_all = [cb["h"] for cb in char_boxes]

    jong_flags: List[Optional[bool]] = [has_jongseong(c) for c in chars]

    for cb, jf in zip(char_boxes, jong_flags):
        if jf is False:
            heights_nojong.append(cb["h"])

    base_h_nojong = _robust_base(heights_nojong, cfg.base_method)
    base_h_all = _robust_base(heights_all, cfg.base_method)

    if base_w is None or base_w <= 0:
        base_w = base_h_all if base_h_all else 1.0

    if base_h_nojong is None or base_h_nojong <= 0:
        base_h_nojong = base_h_all if base_h_all else 1.0
        base_h_note = "no-nojong-found; base_h estimated from all"
    else:
        base_h_note = "base_h estimated from no-jongseong chars"

    slot = 100.0 / n
    slot_x = slot * cfg.w_x
    slot_y = slot * cfg.w_y

    total = 0.0
    details = []

    for idx, (cb, ch, jf) in enumerate(zip(char_boxes, chars, jong_flags), start=1):
        w = cb["w"]
        h = cb["h"]

        rel_x = abs(w - base_w) / base_w if base_w > 0 else 1.0
        sx, zx = _score_from_rel_error(rel_x, slot_x, cfg.tol_ratio_x, cfg.zero_ratio_x, cfg.power)

        if jf is True:
            expected_h = base_h_nojong * cfg.jong_height_ratio
            y_case = "jong"
        else:
            expected_h = base_h_nojong
            y_case = "no_jong_or_unknown"

        rel_y = abs(h - expected_h) / expected_h if expected_h > 0 else 1.0
        sy, zy = _score_from_rel_error(rel_y, slot_y, cfg.tol_ratio_y, cfg.zero_ratio_y, cfg.power)

        s = sx + sy
        if s < 0.0:
            s = 0.0
        elif s > slot:
            s = slot

        total += s
        details.append({
            "index": idx,
            "char": ch,
            "has_jong": jf,
            "bbox_file": cb.get("file"),
            "w": w,
            "h": h,
            "base_w": base_w,
            "base_h_nojong": base_h_nojong,
            "expected_h": expected_h,
            "rel_err_x": rel_x,
            "rel_err_y": rel_y,
            "score_x": sx,
            "score_y": sy,
            "score_char": s,
            "y_case": y_case,
        })

    if total < 0.0:
        total = 0.0
    elif total > 100.0:
        total = 100.0

    return {
        "score": total,
        "count": n,
        "slot": slot,
        "base_w": base_w,
        "base_h_nojong": base_h_nojong,
        "base_h_note": base_h_note,
        "config": cfg.__dict__,
        "details": details,
    }


def score_size_text_distributed(charbbox_text: str, original_text: str, cfg: SizeScoreConfig = SizeScoreConfig()) -> Dict:
    boxes = parse_charbbox_text(charbbox_text, cfg)
    return compute_size_score_distributed(boxes, original_text, cfg)
