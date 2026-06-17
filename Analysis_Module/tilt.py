from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional

from Analysis_Module.common_paths import find_project_root, get_handwriting_dir


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


def compute_center_tilts(centers: List[Tuple[str, float, float]]):
    if len(centers) < 2:
        return []

    centers_sorted = sorted(centers, key=lambda t: t[1])

    tilts = []
    for i in range(len(centers_sorted) - 1):
        ch1, x1, y1 = centers_sorted[i]
        ch2, x2, y2 = centers_sorted[i + 1]

        dx = x2 - x1
        dy = -(y2 - y1)

        tilt_rad = math.atan2(dy, dx)
        tilt_deg = tilt_rad * 180.0 / math.pi

        tilts.append((f"{ch1},{ch2}", tilt_deg))

    return tilts


@dataclass
class TiltResult:
    handwriting_id: str
    center_txt: str
    out_dir: str
    tilt_txt: str
    manifest_path: str


def run_tilt(
    handwriting_id: str,
    project_root: Optional[Path] = None,
    center_txt_path: Optional[str | Path] = None,
) -> TiltResult:
    project_root = project_root or find_project_root(marker_dir="Analysis_Module")
    handwriting_dir = get_handwriting_dir(project_root, handwriting_id)

    if center_txt_path is None:
        center_txt_path = handwriting_dir / "center" / "center.txt"
    else:
        center_txt_path = Path(center_txt_path)

    if not center_txt_path.exists():
        raise FileNotFoundError(f"center.txt not found: {center_txt_path}")

    out_dir = handwriting_dir / "tilt"
    out_dir.mkdir(parents=True, exist_ok=True)

    tilt_txt = out_dir / "tilt.txt"
    manifest_path = out_dir / "manifest.json"
    log_path = out_dir / "log.txt"

    centers = read_centers_char_format(center_txt_path)
    tilts = compute_center_tilts(centers)

    with open(tilt_txt, "w", encoding="utf-8") as f:
        for label, a in tilts:
            f.write(f"{label}: {a:.6f}\n")

    result = TiltResult(
        handwriting_id=handwriting_id,
        center_txt=str(center_txt_path),
        out_dir=str(out_dir),
        tilt_txt=str(tilt_txt),
        manifest_path=str(manifest_path),
    )

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("[OK] tilt\n")
        f.write(f"  centers: {center_txt_path}\n")
        f.write(f"  output : {tilt_txt}\n")

    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Stage: tilt (center-based)")
    ap.add_argument("--handwriting_id", required=True)
    args = ap.parse_args()

    run_tilt(handwriting_id=args.handwriting_id)
