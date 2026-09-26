"""core/fit/units.h: heights, offsets and predictions are logits; variances are squared logits."""
from __future__ import annotations

K_HEIGHT_SCALE = 1.0
K_VARIANCE_SCALE = K_HEIGHT_SCALE * K_HEIGHT_SCALE
# Absolute height guard for candidate proposals, 12 logits: it cannot bind legitimate structure
# but catches split-dependent runaway proposals.
K_HEIGHT_LIMIT_LOGITS = 12.0
K_HEIGHT_LIMIT_UNITS = K_HEIGHT_LIMIT_LOGITS * K_HEIGHT_SCALE
