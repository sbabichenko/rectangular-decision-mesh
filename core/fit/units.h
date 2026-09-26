#pragma once

// Heights, offsets and predictions are logits; variances are squared logits.
inline constexpr double kHeightScale = 1.0;
inline constexpr double kVarianceScale = kHeightScale * kHeightScale;

// Absolute height guard for candidate proposals, 12 logits: it cannot bind legitimate structure
// but catches split-dependent runaway proposals.
inline constexpr double kHeightLimitLogits = 12.0;
inline constexpr double kHeightLimitUnits = kHeightLimitLogits * kHeightScale;
