from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re

@dataclass(frozen=True)
class JamoScoreOnlyConfig:
    total_scale: float = 100.0
    strip_whitespace: bool = True

    w_rel12: float = 1.0
    w_rel23: float = 1.0


_char_header_re = re.compile(r"^\[CHAR\]\s+(?P<name>\S+)\s*$")
_jamo_line_re = re.compile(r"^\[(?P<label>초성|중성|종성)\]\s+(?P<body>.+)$")
_kv_re = re.compile(r"(?P<k>[a-zA-Z0-9_]+)\s*=\s*(?P<v>None|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def _to_float_or_none(s: str) -> Optional[float]:
    if s is None:
        return None
    if s.strip() == "None":
        return None
    return float(s)

def parse_jamo_mvp(text: str) -> List[Dict[str, Any]]:
    lines = [ln.strip() for ln in text.splitlines()]
    chars: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None

    for ln in lines:
        if not ln or ln.startswith("#"):
            continue

        m_char = _char_header_re.match(ln)
        if m_char:
            cur = {"char_id": m_char.group("name")}
            chars.append(cur)
            continue

        if cur is None:
            continue

        m_jamo = _jamo_line_re.match(ln)
        if not m_jamo:
            continue

        label = m_jamo.group("label")
        body = m_jamo.group("body")

        kvs: Dict[str, Optional[float]] = {}
        for m in _kv_re.finditer(body):
            k = m.group("k")
            v = _to_float_or_none(m.group("v"))
            kvs[k] = v

        cur[label] = kvs

    return chars

def _clamp01(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x

def _mean_ignore_none(xs: List[Optional[float]]) -> Optional[float]:
    vals = [x for x in xs if isinstance(x, (int, float))]
    if not vals:
        return None
    return sum(float(v) for v in vals) / len(vals)

def _extract_char_list(original_text: str, strip_whitespace: bool) -> List[str]:
    if not original_text:
        return []
    if strip_whitespace:
        return [c for c in original_text if not c.isspace()]
    return list(original_text)


def score_jamo_score_only(
    jamo_mvp_text: str,
    original_text: str,
    cfg: JamoScoreOnlyConfig = JamoScoreOnlyConfig(),
) -> Dict[str, Any]:
    parsed = parse_jamo_mvp(jamo_mvp_text)
    chars = _extract_char_list(original_text, cfg.strip_whitespace)

    n = len(parsed)
    if len(chars) < n:
        chars = chars + ["?"] * (n - len(chars))
    elif len(chars) > n:
        chars = chars[:n]

    pos_items = []
    shp_items = []

    pos_collect: List[Optional[float]] = []
    shp_collect: List[Optional[float]] = []

    for idx, item in enumerate(parsed):
        ch = chars[idx] if idx < len(chars) else "?"

        def _get_rel(key: str) -> Optional[float]:
            for lab in ("중성", "초성", "종성"):
                d = item.get(lab)
                if isinstance(d, dict) and key in d:
                    return _clamp01(d.get(key))
            return None

        rel12 = _get_rel("rel_score_12")
        rel23 = _get_rel("rel_score_23")

        if rel12 is not None and rel23 is not None:
            pos_char = 0.5 * (rel12 + rel23)
        elif rel12 is not None:
            pos_char = rel12
        else:
            pos_char = None

        cho_shp = _clamp01(item.get("초성", {}).get("area_score") if isinstance(item.get("초성"), dict) else None)
        jung_shp = _clamp01(item.get("중성", {}).get("area_score") if isinstance(item.get("중성"), dict) else None)
        jong_shp = _clamp01(item.get("종성", {}).get("area_score") if isinstance(item.get("종성"), dict) else None)

        pos_items.append({
            "char": ch,
            "char_id": item.get("char_id"),
            "rel12": {"score": rel12},
            "rel23": {"score": rel23},
            "score": pos_char,
        })

        shp_items.append({
            "char": ch,
            "char_id": item.get("char_id"),
            "chosung": {"score": cho_shp},
            "jungsung": {"score": jung_shp},
            "jongsung": {"score": jong_shp},
        })

        pos_collect.append(pos_char)
        shp_collect.extend([cho_shp, jung_shp, jong_shp])

    pos_mean = _mean_ignore_none(pos_collect)
    shp_mean = _mean_ignore_none(shp_collect)

    return {
        "jamo_position": {
            "jamo_position_score": 0.0 if pos_mean is None else cfg.total_scale * pos_mean,
            "items": pos_items,
        },
        "jamo_shape": {
            "jamo_shape_score": 0.0 if shp_mean is None else cfg.total_scale * shp_mean,
            "items": shp_items,
        },
        "meta": {
            "chars_parsed": len(parsed),
            "strip_whitespace": cfg.strip_whitespace,
        },
        "config": cfg.__dict__,
    }