from __future__ import annotations

import json
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


def compute_letter_spacings_with_text(
    centers: List[Tuple[str, float, float]],
    original_text: str,
) -> List[Tuple[str, str, float, bool]]:
    items: List[Tuple[str, str, float, bool]] = []

    if len(centers) < 2:
        return items

    non_space_indices = [idx for idx, ch in enumerate(original_text) if ch != " "]
    if len(non_space_indices) != len(centers):
        raise ValueError(
            "텍스트(공백 제외) 글자 수와 center 개수가 다릅니다: "
            f"text_non_space={len(non_space_indices)}, centers={len(centers)}"
        )

    centers_sorted = sorted(centers, key=lambda t: t[1])

    for i in range(len(centers_sorted) - 1):
        ch1, x1, y1 = centers_sorted[i]
        ch2, x2, y2 = centers_sorted[i + 1]

        idx1 = non_space_indices[i]
        idx2 = non_space_indices[i + 1]

        segment = original_text[idx1 + 1 : idx2 + 1]
        has_space = (" " in segment)

        spacing = float(x2 - x1)
        items.append((ch1, ch2, spacing, has_space))

    return items


@dataclass
class SpaceResult:
    handwriting_id: str
    center_txt: str
    out_dir: str
    space_txt: str
    manifest_path: str


def run_space(
    handwriting_id: str,
    original_text: str,
    project_root: Optional[Path] = None,
    center_txt_path: Optional[str | Path] = None,
) -> SpaceResult:
    project_root = project_root or find_project_root(marker_dir="Analysis_Module")
    handwriting_dir = get_handwriting_dir(project_root, handwriting_id)

    if center_txt_path is None:
        center_txt_path = handwriting_dir / "center" / "center.txt"
    else:
        center_txt_path = Path(center_txt_path)

    if not center_txt_path.exists():
        raise FileNotFoundError(f"center.txt not found: {center_txt_path}")

    out_dir = handwriting_dir / "space"
    out_dir.mkdir(parents=True, exist_ok=True)

    space_txt = out_dir / "space.txt"
    manifest_path = out_dir / "manifest.json"
    log_path = out_dir / "log.txt"

    centers = read_centers_char_format(center_txt_path)
    spacing_items = compute_letter_spacings_with_text(centers=centers, original_text=original_text)

    with open(space_txt, "w", encoding="utf-8") as f:
        for ch1, ch2, spacing, has_space in spacing_items:
            flag = "gap" if has_space else "normal"
            f.write(f"{ch1},{ch2}: {spacing:.2f},{flag}\n")

    result = SpaceResult(
        handwriting_id=handwriting_id,
        center_txt=str(center_txt_path),
        out_dir=str(out_dir),
        space_txt=str(space_txt),
        manifest_path=str(manifest_path),
    )

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("[OK] space\n")
        f.write(f"  centers: {center_txt_path}\n")
        f.write(f"  text   : {original_text}\n")
        f.write(f"  output : {space_txt}\n")

    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Stage: space (center-based)")
    ap.add_argument("--handwriting_id", required=True)
    ap.add_argument("--original_text", required=True)
    args = ap.parse_args()

    run_space(handwriting_id=args.handwriting_id, original_text=args.original_text)
