from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import re
import statistics


_float_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class SpaceScoreConfig:
    gap_ratio: float = 1.30

    tol_ratio: float = 0.10

    zero_ratio: float = 0.40

    power: float = 2.0

    base_method: str = "median"


def parse_space_pairs(text: str) -> List[Tuple[str, float, str]]:
    out: List[Tuple[str, float, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue

        if ":" in line:
            label_part, rest = line.split(":", 1)
            label = label_part.strip()
            rest = rest.strip()
        else:
            label = f"pair_{len(out)}"
            rest = line

        m = _float_re.search(rest)
        if not m:
            continue
        spacing = float(m.group(0))

        tag = "normal"
        if "," in rest:
            tag = rest.split(",")[-1].strip().lower()
        else:
            parts = rest.split()
            if len(parts) >= 2:
                tag = parts[-1].strip().lower()

        if tag not in ("normal", "gap"):
            tag = "normal"

        out.append((label, spacing, tag))
    return out


def _robust_base_normal(normals: List[float], method: str) -> Optional[float]:
    if not normals:
        return None
    if method == "mean":
        return float(sum(normals) / len(normals))
    return float(statistics.median(normals))


def compute_space_score_distributed(
    pairs: List[Tuple[str, float, str]],
    cfg: SpaceScoreConfig = SpaceScoreConfig(),
) -> Dict:
    n = len(pairs)
    if n <= 0:
        return {
            "score": 0.0,
            "count": 0,
            "slot": 0.0,
            "details": [],
            "config": cfg.__dict__,
            "warning": "No space pairs found.",
        }

    if cfg.zero_ratio <= cfg.tol_ratio:
        raise ValueError("zero_ratio must be greater than tol_ratio.")

    slot = 100.0 / n

    normal_vals = [v for _, v, tag in pairs if tag == "normal"]
    base_normal = _robust_base_normal(normal_vals, cfg.base_method)

    if base_normal is None:
        base_normal = _robust_base_normal([v for _, v, _ in pairs], cfg.base_method)
        base_note = "no-normal-found; base estimated from all pairs"
    else:
        base_note = "base estimated from normal pairs"

    assert base_normal is not None and base_normal > 0.0

    total = 0.0
    details = []

    denom = (cfg.zero_ratio - cfg.tol_ratio)

    for label, spacing, tag in pairs:
        expected = base_normal * (cfg.gap_ratio if tag == "gap" else 1.0)

        r = abs(spacing - expected) / expected if expected > 0 else 1.0

        if r <= cfg.tol_ratio:
            s = slot
            z = 0.0
        elif r >= cfg.zero_ratio:
            s = 0.0
            z = 1.0
        else:
            z = (r - cfg.tol_ratio) / denom 
            s = slot * (1.0 - (z ** cfg.power))

        if s < 0.0:
            s = 0.0
        elif s > slot:
            s = slot

        total += s
        details.append({
            "pair": label,
            "tag": tag,
            "spacing": spacing,
            "expected": expected,
            "rel_error": r,
            "z": z,
            "slot": slot,
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
        "base_normal": base_normal,
        "base_note": base_note,
        "config": cfg.__dict__,
        "details": details,
    }


def score_space_text_distributed(space_text: str, cfg: SpaceScoreConfig = SpaceScoreConfig()) -> Dict:
    pairs = parse_space_pairs(space_text)
    return compute_space_score_distributed(pairs, cfg)
