"""Decoded-pixel materialization for Extract."""

from __future__ import annotations

import numpy as np

from pixipix.models import Component, ExtractedFrame, FrameImage

from .analysis import _Analysis


def _materialize_frame_crop(
    analysis: _Analysis,
    component: Component,
    frame: ExtractedFrame,
) -> FrameImage:
    padded = frame.padded_bounds
    crop = np.array(
        analysis.source.pixels[padded.top : padded.bottom, padded.left : padded.right],
        dtype=np.uint8,
        copy=True,
    )
    label_crop = analysis.component_map.labels[
        padded.top : padded.bottom, padded.left : padded.right
    ]
    crop[label_crop != component.discovery_index + 1] = 0
    return FrameImage(frame, crop)
