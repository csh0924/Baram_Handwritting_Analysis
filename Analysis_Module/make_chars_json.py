from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

from Analysis_Module.common_paths import find_project_root, get_handwriting_dir


@dataclass
class MakeCharsJsonConfig:
    img_exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    label_ext: str = ".png"

    cropped_dir_name: str = "char_cropped"
    refined_dir_name: str = "segments_refined"
    refined_labels_subdir: str = "labels_png"

    printed_chars_dir_name: str = "printed_chars"
    printed_segments_dir_name: str = "printed_segments"
    printed_labels_subdir: str = "labels_png"

    out_filename: str = "chars.json"
    store_relative_paths: bool = True


def _list_files(folder: Path, exts: Tuple[str, ...]) -> List[Path]:
    if not folder.exists():
        return []
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: p.name)
    return files


def _path_to_json_str(path: Optional[Path], base: Path, use_rel: bool) -> Optional[str]:
    if path is None:
        return None
    path = Path(path)

    if use_rel:
        try:
            return path.relative_to(base).as_posix()
        except Exception:
            return path.resolve().as_posix()
    else:
        return path.resolve().as_posix()


def _strip_ws_chars(sentence: str) -> List[str]:
    """공백 제거한 문자 리스트"""
    return [c for c in sentence if not c.isspace()]


def make_chars_json(
    handwriting_id: str,
    original_text: str,
    *,
    config: Optional[MakeCharsJsonConfig] = None,
    project_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    if config is None:
        config = MakeCharsJsonConfig()

    if project_root is None:
        project_root = find_project_root(marker_dir="Analysis_Module")
    else:
        project_root = Path(project_root)

    hw_dir = get_handwriting_dir(project_root, handwriting_id)

    base_for_rel = project_root

    cropped_dir = hw_dir / config.cropped_dir_name
    refined_labels_dir = hw_dir / config.refined_dir_name / config.refined_labels_subdir

    printed_chars_dir = hw_dir / config.printed_chars_dir_name
    printed_labels_dir = hw_dir / config.printed_segments_dir_name / config.printed_labels_subdir

    cropped_imgs = _list_files(cropped_dir, config.img_exts)
    refined_labels = _list_files(refined_labels_dir, (config.label_ext,))
    printed_imgs = _list_files(printed_chars_dir, config.img_exts)
    printed_labels = _list_files(printed_labels_dir, (config.label_ext,))

    refined_label_map = {p.stem: p for p in refined_labels}
    printed_label_map = {p.stem: p for p in printed_labels}
    printed_img_map = {p.stem: p for p in printed_imgs}

    chars = _strip_ws_chars(original_text)
    n = min(len(chars), len(cropped_imgs))

    items: List[Dict[str, Any]] = []

    for i in range(n):
        ch = chars[i]
        cropped_img = cropped_imgs[i]
        stem = cropped_img.stem

        item = {
            "char": ch,
            "handwritten": {
                "image": _path_to_json_str(cropped_img, base_for_rel, config.store_relative_paths),
                "label_png": _path_to_json_str(
                    refined_label_map.get(stem), base_for_rel, config.store_relative_paths
                ) if stem in refined_label_map else None,
            },
            "printed": {
                "image": _path_to_json_str(
                    printed_img_map.get(stem), base_for_rel, config.store_relative_paths
                ) if stem in printed_img_map else None,
                "label_png": _path_to_json_str(
                    printed_label_map.get(stem), base_for_rel, config.store_relative_paths
                ) if stem in printed_label_map else None,
            },
        }

        items.append(item)

    out_obj: Dict[str, Any] = {
        "handwriting_id": handwriting_id,
        "original_text": original_text,
        "counts": {
            "chars_in_text": len(chars),
            "paired_items": len(items),
        },
        "items": items,
        "config": asdict(config),
    }

    out_path = hw_dir / config.out_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    return out_obj


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser("make chars.json (handwritten + printed)")
    ap.add_argument("--handwriting_id", required=True)
    ap.add_argument("--original_text", required=True)
    ap.add_argument("--abs_paths", action="store_true")

    args = ap.parse_args()

    cfg = MakeCharsJsonConfig(
        store_relative_paths=(not args.abs_paths),
    )

    res = make_chars_json(
        handwriting_id=args.handwriting_id,
        original_text=args.original_text,
        config=cfg,
    )

    print(f"[OK] saved: Analysis_Data/{args.handwriting_id}/chars.json")
    print("paired_items:", res["counts"]["paired_items"])
