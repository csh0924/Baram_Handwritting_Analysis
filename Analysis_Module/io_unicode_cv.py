from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import cv2


def imwrite_unicode(path: str | Path, img: np.ndarray) -> bool:
    path = str(path)
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True


def imread_unicode(path: str | Path, flags=cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    try:
        path = str(path)
        with open(path, "rb") as f:
            data = f.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, flags)
        return img
    except Exception as e:
        print("[imread_unicode ERROR]", e)
        return None
