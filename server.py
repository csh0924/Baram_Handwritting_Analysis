from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from flask import Flask, request, jsonify


def setup_project_path() -> Path:
    current = Path.cwd()
    while current != current.parent and not (current / "Analysis_Module").exists():
        current = current.parent
    if not (current / "Analysis_Module").exists():
        raise RuntimeError("Could not find project_root containing 'Analysis_Module' directory.")
    return current


PROJECT_ROOT = setup_project_path()
DATA_ROOT = PROJECT_ROOT / "Analysis_Data"

app = Flask(__name__)


def _sanitize_id(handwriting_id: str) -> str:
    s = handwriting_id.strip()
    if not re.fullmatch(r"[0-9A-Za-z_-]+", s):
        raise ValueError("Invalid handwriting_id")
    return s


def _load_json(path: Path, name: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_analysis_error_payload(stdout: str | None, stderr: str | None) -> Optional[Dict[str, Any]]:
    marker = "__ANALYSIS_ERROR__"

    def _find_and_parse(text: str | None) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        idx = text.find(marker)
        if idx == -1:
            return None
        tail = text[idx + len(marker):].strip()

        try:
            return json.loads(tail)
        except Exception:
            first_line = tail.splitlines()[0].strip() if tail.splitlines() else tail
            try:
                return json.loads(first_line)
            except Exception:
                return {
                    "type": "UnknownErrorPayload",
                    "message": "Failed to parse __ANALYSIS_ERROR__ payload as JSON",
                    "raw": tail[:4000],
                }

    return _find_and_parse(stdout) or _find_and_parse(stderr)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        handwriting_id = _sanitize_id(request.form.get("handwriting_id", ""))
        original_text = request.form.get("original_text", "").strip()
        font = request.form.get("font", "").strip()

        if not handwriting_id or not original_text or not font:
            return jsonify({"error": "Missing required fields"}), 400

        if "image" not in request.files:
            return jsonify({"error": "image file is required"}), 400

        image_file = request.files["image"]

        image_dir = DATA_ROOT / handwriting_id
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{handwriting_id}.png"
        image_file.save(str(image_path))

        cmd = [
            sys.executable, "-m", "Analysis_Module.main",
            "--handwriting_id", handwriting_id,
            "--original_text", original_text,
            "--image_path", str(image_path),
            "--font", font,
        ]

        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        log_path = image_dir / "run.log"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(proc.stdout or "")
            f.write("\n\n--- STDERR ---\n\n")
            f.write(proc.stderr or "")

        if proc.returncode != 0:
            payload = _extract_analysis_error_payload(proc.stdout, proc.stderr)

            if payload and payload.get("type") == "CenterDetectionError":
                return jsonify({
                    "error": "center detection failed",
                    "detail": payload,
                }), 422

            return jsonify({
                "error": "analysis failed",
                "detail": payload,
            }), 500

        analysed_path = image_dir / "analysed.json"
        chars_path = image_dir / "chars.json"

        analysed_json = _load_json(analysed_path, "analysed.json")
        chars_json = _load_json(chars_path, "chars.json")

        return jsonify({
            "analysed": analysed_json,
            "chars": chars_json,
        }), 200

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        return jsonify({"error": "internal server error"}), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
    )
