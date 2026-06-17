from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import re

_float_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class TiltScoreConfig:
    free_deg: float = 1.0     
    zero_deg: float = 30.0 
    power: float = 0.85        


def parse_tilt_pairs(text: str) -> List[Tuple[str, float]]:
    pairs: List[Tuple[str, float]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue

        m = _float_re.search(line)
        if not m:
            continue

        angle = float(m.group(0))
        label = line[:m.start()].strip().rstrip(":").strip()
        if not label:
            label = f"pair_{len(pairs)}"
        pairs.append((label, angle))
    return pairs


def compute_tilt_score_distributed(
    pairs: List[Tuple[str, float]],
    cfg: TiltScoreConfig = TiltScoreConfig(),
) -> Dict:
    n = len(pairs)
    if n <= 0:
        return {
            "score": 0.0,
            "count": 0,
            "slot": 0.0,
            "details": [],
            "config": cfg.__dict__,
            "warning": "No tilt pairs found.",
        }

    if cfg.zero_deg <= cfg.free_deg:
        raise ValueError("zero_deg must be greater than free_deg.")

    slot = 100.0 / n
    total = 0.0
    details = []

    denom = (cfg.zero_deg - cfg.free_deg)

    for label, a in pairs:
        abs_a = abs(a)

        if abs_a <= cfg.free_deg:
            s = slot
            r = 0.0
        elif abs_a >= cfg.zero_deg:
            s = 0.0
            r = 1.0
        else:
            r = (abs_a - cfg.free_deg) / denom
            s = slot * (1.0 - (r ** cfg.power))

        if s < 0.0:
            s = 0.0
        elif s > slot:
            s = slot

        total += s
        details.append({
            "pair": label,
            "angle_deg": a,
            "abs_angle_deg": abs_a,
            "slot": slot,
            "ratio_r": r,
            "score_pair": s,
        })

    if total < 0.0:
        total = 0.0
    elif total > 100.0:
        total = 100.0

    return {
        "score": total,
        "count": n,
        "slot": slot,
        "details": details,
        "config": cfg.__dict__,
    }


def score_tilt_text_distributed(tilt_text: str, cfg: TiltScoreConfig = TiltScoreConfig()) -> Dict:
    pairs = parse_tilt_pairs(tilt_text)
    return compute_tilt_score_distributed(pairs, cfg)
