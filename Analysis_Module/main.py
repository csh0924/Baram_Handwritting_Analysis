from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

from Analysis_Module.normalize_1 import run_normalize_1, Normalize1Config
from Analysis_Module.center import run_center, CenterConfig
from Analysis_Module.normalize_2 import run_normalize_2, Normalize2Config
from Analysis_Module.tilt import run_tilt
from Analysis_Module.space import run_space

from Analysis_Module.char_split import run_char_split, CharSplitConfig
from Analysis_Module.char_bbox import run_char_bbox, CharBBoxConfig

from Analysis_Module.fpn_segment import run_fpn_segment, FPNSegmentConfig
from Analysis_Module.refine_segments import run_refine_segments, RefineSegmentsConfig
from Analysis_Module.generate_printed_chars import run_printed_chars, PrintedCharsConfig

from Analysis_Module.jamo_eval import run_jamo_eval, JamoEvalConfig
from Analysis_Module.make_score import make_score

from Analysis_Module.make_json import make_json
from Analysis_Module.make_chars_json import make_chars_json

from Analysis_Module.errors import CenterDetectionError


def run_pipeline(
    handwriting_id: str,
    original_text: str,
    image_path: Path,
    cfg_norm: Normalize1Config,
    cfg_center: CenterConfig,
    cfg_char_split: CharSplitConfig,
    cfg_char_bbox: CharBBoxConfig,
    cfg_fpn_hand: FPNSegmentConfig,
    cfg_printed: PrintedCharsConfig,
    cfg_fpn_printed: FPNSegmentConfig,
    cfg_refine: RefineSegmentsConfig,
    cfg_jamo: JamoEvalConfig,
) -> Dict[str, Any]:

    norm_result = run_normalize_1(
        handwriting_id=handwriting_id,
        config=cfg_norm,
    )

    center_result = run_center(
        handwriting_id=handwriting_id,
        original_text=original_text,
        normalized_image_path=Path(norm_result["normalized_image_path"])
        if isinstance(norm_result, dict) and norm_result.get("normalized_image_path")
        else None,
        config=cfg_center,
    )

    n2_result = run_normalize_2(handwriting_id=handwriting_id, config=Normalize2Config())
    tilt_result = run_tilt(handwriting_id)
    space_result = run_space(handwriting_id, original_text)

    char_result = run_char_split(handwriting_id=handwriting_id, config=cfg_char_split)
    
    charbbox_result = run_char_bbox(
        handwriting_id=handwriting_id,
        config=cfg_char_bbox,
        roi_map=char_result.roi_map,
    )

    fpn_hand_result = run_fpn_segment(
        handwriting_id=handwriting_id,
        original_text=original_text,
        config=cfg_fpn_hand,
        fonts=False,
    )

    printed_result = run_printed_chars(
        handwriting_id=handwriting_id,
        original_text=original_text,
        config=cfg_printed,
    )

    fpn_printed_result = run_fpn_segment(
        handwriting_id=handwriting_id,
        original_text=original_text,
        config=cfg_fpn_printed,
        fonts=True,
    )

    refine_result = run_refine_segments(
        handwriting_id=handwriting_id,
        config=cfg_refine,
    )

    jamo_result = run_jamo_eval(
        handwriting_id=handwriting_id,
        config=cfg_jamo,
    )

    return {
        "normalize_1": norm_result,
        "center": center_result,
        "normalize_2": n2_result,
        "tilt": tilt_result,
        "space": space_result,
        "char_split": char_result,
        "char_bbox": charbbox_result,
        "fpn_handwriting": fpn_hand_result,
        "printed_chars": printed_result,
        "fpn_printed": fpn_printed_result,
        "refine_segments": refine_result,
        "jamo_eval": jamo_result,
    }


def main():
    parser = argparse.ArgumentParser(description="Baram Handwriting Analysis Pipeline")

    parser.add_argument("--handwriting_id", required=True)
    parser.add_argument("--original_text", required=True)
    parser.add_argument("--image_path", required=True)

    parser.add_argument("--target_stroke_px", type=int, default=10)
    parser.add_argument("--binarize_method", choices=["otsu", "adaptive"], default="otsu")
    parser.add_argument("--no_invert", action="store_true")
    parser.add_argument("--open_kernel", type=int, default=3)

    parser.add_argument("--center_thr", type=float, default=0.4)
    parser.add_argument("--center_min_area", type=int, default=30)
    parser.add_argument("--text_threshold", type=float, default=0.5)
    parser.add_argument("--link_threshold", type=float, default=0.1)
    parser.add_argument("--low_text", type=float, default=0.2)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--font", default=None, help="Analysis_Module/fonts 아래 폰트 파일명 (None이면 자동선택)")

    parser.add_argument(
        "--fpn_backbone",
        default="resnet34",
        choices=["resnet18", "resnet34", "resnet50", "resnext50"],
    )
    parser.add_argument("--fpn_cpu", action="store_true")
    parser.add_argument("--fpn_no_backbone_check", action="store_true")
    parser.add_argument("--fpn_size", type=int, default=256)

    parser.add_argument("--refine_out_subdir", default="segments_refined")
    parser.add_argument("--refine_ink_thr", type=int, default=160)
    parser.add_argument("--refine_max_dist", type=float, default=8.0)
    parser.add_argument("--refine_suppress_min_ratio", type=float, default=0.30)
    parser.add_argument("--refine_suppress_min_pixels", type=int, default=20)
    parser.add_argument("--refine_save_masked", action="store_true")

    parser.add_argument("--jamo_sigma_centroid", type=float, default=15.0)
    parser.add_argument("--jamo_dilate_r", type=int, default=3)
    parser.add_argument("--jamo_sigma_angle_deg", type=float, default=20.0)
    parser.add_argument("--jamo_sigma_len", type=float, default=20.0)
    parser.add_argument("--jamo_limit", type=int, default=None, help="평가 개수 제한(디버그용), None이면 전체")

    args = parser.parse_args()

    cfg_norm = Normalize1Config(
        target_stroke_px=args.target_stroke_px,
        binarize_method=args.binarize_method,
        invert=(not args.no_invert),
        open_kernel=args.open_kernel,
    )

    cfg_center = CenterConfig(
        center_thr=args.center_thr,
        center_min_area=args.center_min_area,
        text_threshold=args.text_threshold,
        link_threshold=args.link_threshold,
        low_text=args.low_text,
        use_cuda=(not args.cpu),
    )

    cfg_char_split = CharSplitConfig()
    cfg_char_bbox = CharBBoxConfig()

    cfg_fpn_hand = FPNSegmentConfig(
        backbone=args.fpn_backbone,
        force_cpu=args.fpn_cpu,
        check_backbone_match=(not args.fpn_no_backbone_check),
        size=(args.fpn_size, args.fpn_size),
    )

    cfg_printed = PrintedCharsConfig(
        font_name=args.font,
    )

    cfg_fpn_printed = FPNSegmentConfig(
        backbone=args.fpn_backbone,
        force_cpu=args.fpn_cpu,
        check_backbone_match=(not args.fpn_no_backbone_check),
        size=(args.fpn_size, args.fpn_size),
    )

    cfg_refine = RefineSegmentsConfig(
        out_subdir_name=args.refine_out_subdir,
        ink_thr=args.refine_ink_thr,
        max_dist=args.refine_max_dist,
        suppress_min_ratio=args.refine_suppress_min_ratio,
        suppress_min_pixels=args.refine_suppress_min_pixels,
        save_masked_vis=args.refine_save_masked,
    )

    cfg_jamo = JamoEvalConfig(
        sigma_centroid=args.jamo_sigma_centroid,
        dilate_r=args.jamo_dilate_r,
        sigma_angle_deg=args.jamo_sigma_angle_deg,
        sigma_len=args.jamo_sigma_len,
    )

    try:
        outputs = run_pipeline(
            handwriting_id=args.handwriting_id,
            original_text=args.original_text,
            image_path=Path(args.image_path),
            cfg_norm=cfg_norm,
            cfg_center=cfg_center,
            cfg_char_split=cfg_char_split,
            cfg_char_bbox=cfg_char_bbox,
            cfg_fpn_hand=cfg_fpn_hand,
            cfg_printed=cfg_printed,
            cfg_fpn_printed=cfg_fpn_printed,
            cfg_refine=cfg_refine,
            cfg_jamo=cfg_jamo,
        )
    except CenterDetectionError as e:
        payload = {
            "type": "CenterDetectionError",
            "message": str(e),
            "expected": getattr(e, "expected", None),
            "detected": getattr(e, "detected", None),
            "raw_detected": getattr(e, "raw_detected", None),
            "handwriting_id": args.handwriting_id,
            "original_text": args.original_text,
        }
        print("__ANALYSIS_ERROR__" + json.dumps(payload, ensure_ascii=False), file=sys.stdout)
        raise SystemExit(2)

    print("\n[PIPELINE RESULT]")
    for stage, result in outputs.items():
        print(f"\n[{stage}]")
        print(result)

    print("\n[MAKE SCORE]")
    try:
        score_result = make_score(
            handwriting_id=args.handwriting_id,
            original_text=args.original_text,
            save=True,
        )
        print("Final score:", score_result["final_score"]["score"])
        print("Score file saved to:", f"Analysis_Data/{args.handwriting_id}/scores/scores.json")
    except Exception as e:
        print("[WARN] make_score failed:", e)

    print("\n[MAKE JSON]")
    try:
        analysed = make_json(
            handwriting_id=args.handwriting_id,
            sentence=args.original_text,
            font=args.font,
            save=True,
        )
        print("Analysed file saved to:", f"Analysis_Data/{args.handwriting_id}/analysed.json")
        print("final_score in analysed.json:", analysed.get("final_score"))
    except Exception as e:
        print("[WARN] build_analysed_json failed:", e)

    print("\n[MAKE CHARS JSON]")
    try:
        chars_obj = make_chars_json(
            handwriting_id=args.handwriting_id,
            original_text=args.original_text,
        )
        print("Chars file saved to:", f"Analysis_Data/{args.handwriting_id}/chars.json")
        print("paired_items in chars.json:", chars_obj.get("counts", {}).get("paired_items"))
    except Exception as e:
        print("[WARN] make_chars_json failed:", e)


if __name__ == "__main__":
    main()