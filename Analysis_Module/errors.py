from __future__ import annotations

class CenterDetectionError(Exception):
    def __init__(self, message: str, *, expected: int, detected: int, raw_detected: int | None = None):
        super().__init__(message)
        self.expected = expected
        self.detected = detected
        self.raw_detected = raw_detected
