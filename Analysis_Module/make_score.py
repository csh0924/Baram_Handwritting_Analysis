from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

from Analysis_Module.common_paths import find_project_root, get_handwriting_dir

from Analysis_Module.score_tilt import score_tilt_text_distributed
from Analysis_Module.score_space import score_space_text_distributed
from Analysis_Module.score_size import score_size_text_distributed

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


@dataclass(frozen=True)
class ScoreInputPaths:
    def tilt(self, hw_dir: Path) -> Path:
        return hw_dir / "tilt" / "tilt.txt"

    def space(self, hw_dir: Path) -> Path:
        return hw_dir / "space" / "space.txt"

    def charbbox(self, hw_dir: Path) -> Path:
        return hw_dir / "char_bbox" / "charbbox.txt"

    def jamo(self, hw_dir: Path) -> Path:
        return hw_dir / "jamo_eval" / "jamo_mvp.txt"


def _safe_get_score(obj: Any) -> Optional[float]:
    if not isinstance(obj, dict):
        return None
    if "error" in obj:
        return None
    v = obj.get("score")
    return float(v) if isinstance(v, (int, float)) else None


def _compute_jamo_total_from_score_only(jamo_obj: Dict[str, Any]) -> Optional[float]:
    if not isinstance(jamo_obj, dict) or "error" in jamo_obj:
        return None

    try:
        pos = jamo_obj["jamo_position"]["jamo_position_score"]
        shp = jamo_obj["jamo_shape"]["jamo_shape_score"]
        if not isinstance(pos, (int, float)) or not isinstance(shp, (int, float)):
            return None
        total = (float(pos) + float(shp)) / 2.0
        return max(0.0, min(100.0, total))
    except Exception:
        return None


def compute_final_score(
    tilt: Optional[float],
    space: Optional[float],
    size: Optional[float],
    jamo_total: Optional[float],
    weights: Dict[str, float],
) -> Dict[str, Any]:
    items = [("tilt", tilt), ("space", space), ("size", size), ("jamo", jamo_total)]
    valid = [(k, v) for k, v in items if isinstance(v, (int, float))]

    if not valid:
        return {
            "score": 0.0,
            "used_weights": {},
            "components": {},
            "warning": "No valid component scores to compute final score.",
        }

    ws = {k: max(0.0, float(weights.get(k, 0.0))) for k, _ in valid}
    w_sum = sum(ws.values())

    if w_sum <= 0.0:
        eq = 1.0 / len(valid)
        ws = {k: eq for k, _ in valid}
    else:
        ws = {k: w / w_sum for k, w in ws.items()}

    final = 0.0
    comps = {}
    for k, v in valid:
        w = ws[k]
        final += w * float(v)
        comps[k] = {"score": float(v), "weight": w, "contribution": w * float(v)}

    final = max(0.0, min(100.0, final))
    return {"score": final, "used_weights": ws, "components": comps}


def make_score(
    handwriting_id: str,
    original_text: str,
    save: bool = True,
    weights: Dict[str, float] | None = None,
    paths: ScoreInputPaths | None = None,
) -> Dict[str, Any]:
    if weights is None:
        weights = {"tilt": 0.25, "space": 0.25, "size": 0.25, "jamo": 0.25}
    paths = paths or ScoreInputPaths()

    project_root = find_project_root()
    hw_dir = get_handwriting_dir(project_root, handwriting_id)

    p_tilt = paths.tilt(hw_dir)
    p_space = paths.space(hw_dir)
    p_charbbox = paths.charbbox(hw_dir)
    p_jamo = paths.jamo(hw_dir)

    out_dir = hw_dir / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scores.json"

    results: Dict[str, Any] = {
        "handwriting_id": handwriting_id,
        "original_text": original_text,
        "paths": {
            "tilt": str(p_tilt),
            "space": str(p_space),
            "charbbox": str(p_charbbox),
            "jamo": str(p_jamo),
        },
        "weights": weights,
    }

    try:
        tilt_text = read_text_file(p_tilt)
        results["tilt"] = score_tilt_text_distributed(tilt_text)
    except Exception as e:
        results["tilt"] = {"error": str(e)}

    try:
        space_text = read_text_file(p_space)
        results["space"] = score_space_text_distributed(space_text)
    except Exception as e:
        results["space"] = {"error": str(e)}

    try:
        charbbox_text = read_text_file(p_charbbox)
        results["size"] = score_size_text_distributed(charbbox_text, original_text)
    except Exception as e:
        results["size"] = {"error": str(e)}

    try:
        jamo_text = read_text_file(p_jamo)
        results["jamo"] = score_jamo_score_only(jamo_text, original_text)
    except Exception as e:
        results["jamo"] = {"error": str(e)}

    tilt_score = _safe_get_score(results.get("tilt"))
    space_score = _safe_get_score(results.get("space"))
    size_score = _safe_get_score(results.get("size"))
    jamo_total = _compute_jamo_total_from_score_only(results.get("jamo", {}))

    results["jamo_total"] = jamo_total
    results["final_score"] = compute_final_score(
        tilt=tilt_score,
        space=space_score,
        size=size_score,
        jamo_total=jamo_total,
        weights=weights,
    )

    if save:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    res = make_score(args.id, args.text, save=True)
    print(f"[OK] saved: Analysis_Data/{args.id}/scores/scores.json")
    print("final_score:", res["final_score"]["score"])