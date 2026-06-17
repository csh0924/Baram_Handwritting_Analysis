from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from Analysis_Module.common_paths import find_project_root, get_handwriting_dir
from Analysis_Module.score_jamo import score_jamo_score_only


def read_text_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    for enc in ("utf-8", "utf-8-sig", "cp949"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue

    return path.read_text(encoding="utf-8", errors="replace")


def read_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_file(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _strip_ws_chars(sentence: str) -> List[str]:
    return [c for c in sentence if not c.isspace()]


_center_patterns = [
    re.compile(r"^(?P<char>.)\s*[:]\s*(?P<x>-?\d+(?:\.\d+)?)\s*[, ]\s*(?P<y>-?\d+(?:\.\d+)?)\s*$"),
    re.compile(r"^\[(?P<char>.)\]\s*x\s*=\s*(?P<x>-?\d+(?:\.\d+)?)\s*y\s*=\s*(?P<y>-?\d+(?:\.\d+)?)\s*$"),
    re.compile(r"^char\s*=\s*(?P<char>.)\s*x\s*=\s*(?P<x>-?\d+(?:\.\d+)?)\s*y\s*=\s*(?P<y>-?\d+(?:\.\d+)?)\s*$"),
]


def parse_center_txt(text: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        for pat in _center_patterns:
            m = pat.match(ln)
            if m:
                items.append({
                    "char": m.group("char"),
                    "x": float(m.group("x")),
                    "y": float(m.group("y")),
                })
                break
    return items


_tilt_re = re.compile(r"^(?P<c1>.)\s*,\s*(?P<c2>.)\s*:\s*(?P<v>-?\d+(?:\.\d+)?)\s*$")


def parse_tilt_txt(text: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        m = _tilt_re.match(ln)
        if not m:
            continue
        items.append({
            "char1": m.group("c1"),
            "char2": m.group("c2"),
            "tilt": float(m.group("v")),
        })
    return items


_space_re = re.compile(
    r"^(?P<c1>.)\s*,\s*(?P<c2>.)\s*:\s*(?P<v>-?\d+(?:\.\d+)?)\s*,\s*(?P<tag>\w+)\s*$",
    re.IGNORECASE,
)


def parse_space_txt(text: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        m = _space_re.match(ln)
        if not m:
            continue
        tag = m.group("tag").lower()
        items.append({
            "char1": m.group("c1"),
            "char2": m.group("c2"),
            "space": float(m.group("v")),
            "gap": (tag == "gap"),
        })
    return items


_file_re = re.compile(r"^file\s*=\s*(?P<fname>\S+)\s*$")
_xyxy_re = re.compile(r"xyxy\s*=\s*\(\s*(?P<x1>\d+)\s*,\s*(?P<y1>\d+)\s*,\s*(?P<x2>\d+)\s*,\s*(?P<y2>\d+)\s*\)\s*$")


def parse_charbbox_txt(text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    cur_file: Optional[str] = None

    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#") or ln.startswith("-"):
            continue

        m1 = _file_re.match(ln)
        if m1:
            cur_file = m1.group("fname")
            continue

        m2 = _xyxy_re.search(ln)
        if m2 and cur_file is not None:
            out.append({
                "file": cur_file,
                "x1": int(m2.group("x1")),
                "y1": int(m2.group("y1")),
                "x2": int(m2.group("x2")),
                "y2": int(m2.group("y2")),
            })
            cur_file = None

    return out


def attach_char_and_compute_rel_size(
    bbox_items: List[Dict[str, Any]],
    sentence: str,
) -> List[Dict[str, Any]]:
    chars = _strip_ws_chars(sentence)
    n = min(len(bbox_items), len(chars))

    areas: List[int] = []
    for i in range(n):
        it = bbox_items[i]
        w = max(1, int(it["x2"]) - int(it["x1"]))
        h = max(1, int(it["y2"]) - int(it["y1"]))
        areas.append(w * h)

    if areas:
        s = sorted(areas)
        med = float(s[len(s) // 2])
        if med <= 0:
            med = float(sum(areas) / len(areas))
            if med <= 0:
                med = 1.0
    else:
        med = 1.0

    out: List[Dict[str, Any]] = []
    for i in range(n):
        it = bbox_items[i]
        ch = chars[i]
        w = max(1, int(it["x2"]) - int(it["x1"]))
        h = max(1, int(it["y2"]) - int(it["y1"]))
        rel = float((w * h) / med)

        out.append({
            "char": ch,
            "x1": int(it["x1"]), "y1": int(it["y1"]),
            "x2": int(it["x2"]), "y2": int(it["y2"]),
            "size": rel,
        })
    return out


def load_jamo_bboxes(jamo_bboxes_path: Path) -> Dict[str, Dict[str, Optional[Dict[str, int]]]]:
    obj = read_json_file(jamo_bboxes_path)
    items = obj.get("items", [])
    out: Dict[str, Dict[str, Optional[Dict[str, int]]]] = {}

    if isinstance(items, list):
        for it in items:
            stem = it.get("stem")
            bb = it.get("bboxes", {})
            if isinstance(stem, str) and isinstance(bb, dict):
                out[stem] = {
                    "chosung": bb.get("chosung"),
                    "jungsung": bb.get("jungsung"),
                    "jongsung": bb.get("jongsung"),
                }
    return out


def _add_bbox_to_jamo_item(
    item: Dict[str, Any],
    bboxes_by_stem: Dict[str, Dict[str, Optional[Dict[str, int]]]],
) -> Dict[str, Any]:
    char_id = item.get("char_id")  
    bb = bboxes_by_stem.get(char_id, {}) if isinstance(char_id, str) else {}

    for key in ("chosung", "jungsung", "jongsung"):
        if key in item and isinstance(item[key], dict):
            bbox = bb.get(key)
            if bbox is not None:
                item[key]["bbox"] = bbox

    item.pop("char_id", None)
    return item


def merge_jamo_score_with_bboxes(
    jamo_score_obj: Dict[str, Any],
    bboxes_by_stem: Dict[str, Dict[str, Optional[Dict[str, int]]]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    jp = jamo_score_obj.get("jamo_position", {})
    js = jamo_score_obj.get("jamo_shape", {})

    pos_items: List[Dict[str, Any]] = []
    for it in jp.get("items", []) if isinstance(jp.get("items"), list) else []:
        if isinstance(it, dict):
            pos_items.append(_add_bbox_to_jamo_item(dict(it), bboxes_by_stem))

    shp_items: List[Dict[str, Any]] = []
    for it in js.get("items", []) if isinstance(js.get("items"), list) else []:
        if isinstance(it, dict):
            it2 = dict(it)
            it2.pop("char_id", None)
            shp_items.append(it2)

    jamo_position_out = {
        "jamo_position_score": float(jp.get("jamo_position_score", 0.0) or 0.0),
        "items": pos_items,
    }
    jamo_shape_out = {
        "jamo_shape_score": float(js.get("jamo_shape_score", 0.0) or 0.0),
        "items": shp_items,
    }
    return jamo_position_out, jamo_shape_out


def make_json(
    handwriting_id: str,
    sentence: str,
    *,
    font: Optional[str] = None,
    save: bool = True,
) -> Dict[str, Any]:
    project_root = find_project_root(marker_dir="Analysis_Module")
    hw_dir = get_handwriting_dir(project_root, handwriting_id)

    p_center = hw_dir / "center" / "center.txt"
    p_tilt = hw_dir / "tilt" / "tilt.txt"
    p_space = hw_dir / "space" / "space.txt"
    p_charbbox = hw_dir / "char_bbox" / "charbbox.txt"

    p_scores = hw_dir / "scores" / "scores.json"

    p_jamo_mvp = hw_dir / "jamo_eval" / "jamo_mvp.txt"
    p_jamo_bboxes = hw_dir / "jamo_bboxes" / "jamo_bboxes.json"

    scores_obj: Dict[str, Any] = {}
    if p_scores.exists():
        try:
            scores_obj = read_json_file(p_scores)
        except Exception:
            scores_obj = {}

    final_score = 0.0
    try:
        final_score = float(scores_obj.get("final_score", {}).get("score", 0.0))
    except Exception:
        final_score = 0.0

    tilt_score = scores_obj.get("tilt", {}).get("score")
    space_score = scores_obj.get("space", {}).get("score")
    size_score = scores_obj.get("size", {}).get("score")

    center_items: List[Dict[str, Any]] = []
    if p_center.exists():
        try:
            center_items = parse_center_txt(read_text_file(p_center))
        except Exception:
            center_items = []

    tilt_items: List[Dict[str, Any]] = []
    if p_tilt.exists():
        try:
            tilt_items = parse_tilt_txt(read_text_file(p_tilt))
        except Exception:
            tilt_items = []

    space_items: List[Dict[str, Any]] = []
    if p_space.exists():
        try:
            space_items = parse_space_txt(read_text_file(p_space))
        except Exception:
            space_items = []

    size_items: List[Dict[str, Any]] = []
    if p_charbbox.exists():
        try:
            raw_bbox = parse_charbbox_txt(read_text_file(p_charbbox))
            size_items = attach_char_and_compute_rel_size(raw_bbox, sentence)
        except Exception:
            size_items = []

    jamo_position_out: Dict[str, Any] = {"jamo_position_score": 0.0, "items": []}
    jamo_shape_out: Dict[str, Any] = {"jamo_shape_score": 0.0, "items": []}

    if p_jamo_mvp.exists():
        try:
            jamo_text = read_text_file(p_jamo_mvp)
            jamo_score_obj = score_jamo_score_only(jamo_text, sentence)

            if p_jamo_bboxes.exists():
                bboxes_by_stem = load_jamo_bboxes(p_jamo_bboxes)
                jamo_position_out, jamo_shape_out = merge_jamo_score_with_bboxes(jamo_score_obj, bboxes_by_stem)
            else:
                jp = jamo_score_obj.get("jamo_position", {})
                js = jamo_score_obj.get("jamo_shape", {})
                jamo_position_out = {
                    "jamo_position_score": float(jp.get("jamo_position_score", 0.0) or 0.0),
                    "items": jp.get("items", []) if isinstance(jp.get("items"), list) else [],
                }
                jamo_shape_out = {
                    "jamo_shape_score": float(js.get("jamo_shape_score", 0.0) or 0.0),
                    "items": js.get("items", []) if isinstance(js.get("items"), list) else [],
                }
        except Exception:
            pass

    analysed: Dict[str, Any] = {
        "handwrite_id": handwriting_id,
        "sentence": sentence,
        "font": font,
        "final_score": float(final_score),
        "analysed": {
            "center": {
                "items": center_items
            },
            "tilt": {
                "tilt_score": float(tilt_score) if isinstance(tilt_score, (int, float)) else 0.0,
                "items": tilt_items
            },
            "space": {
                "space_score": float(space_score) if isinstance(space_score, (int, float)) else 0.0,
                "items": space_items
            },
            "size": {
                "size_score": float(size_score) if isinstance(size_score, (int, float)) else 0.0,
                "items": size_items
            },
            "jamo_position": jamo_position_out,
            "jamo_shape": jamo_shape_out,
        }
    }

    out_path = hw_dir / "analysed.json"
    if save:
        write_json_file(out_path, analysed)

    return analysed


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser("make analysed.json")
    ap.add_argument("--handwriting_id", required=True)
    ap.add_argument("--sentence", required=True)
    ap.add_argument("--font", default=None)
    args = ap.parse_args()

    res = make_json(
        handwriting_id=args.handwriting_id,
        sentence=args.sentence,
        font=args.font,
        save=True,
    )
    print(f"[OK] saved: Analysis_Data/{args.handwriting_id}/analysed.json")
    print("final_score:", res.get("final_score"))
