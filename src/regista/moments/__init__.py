"""Phase 2 moment detectors: causal, streaming, evidence-backed tactical alerts."""

from regista.moments.detectors import DetectorConfig, Moment, detect_moments
from regista.moments.engine import StreamConfig, stream_windows

__all__ = ["DetectorConfig", "Moment", "StreamConfig", "detect_moments", "stream_windows"]
