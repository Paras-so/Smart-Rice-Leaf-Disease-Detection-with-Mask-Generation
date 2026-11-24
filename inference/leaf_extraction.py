"""Automatic leaf-region proposal; image-specific review is still required."""
import cv2
import numpy as np


def propose_leaf_mask(image):
    rgb = np.asarray(image)
    # Work at a bounded resolution, then restore discrete labels with nearest neighbor.
    height, width = rgb.shape[:2]
    scale = min(1.0, 1024 / max(height, width))
    small = cv2.resize(rgb, (max(1, round(width * scale)), max(1, round(height * scale))))
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    candidate = ((h >= 15) & (h <= 100) & (s >= 40) & (v >= 30)).astype(np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(candidate)
    cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
    mask = cv2.resize(filled, (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    flags = []
    if not mask.any():
        flags.append('No leaf region detected.')
    if candidate.mean() > .35:
        flags.append('Vegetation covers much of the scene; background leaves or grass may be included.')
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        flags.append('The mask reaches the image edge; check cropped leaves and background carefully.')
    return dict(mask=mask, validated=False, method='color_support_with_closed_contours_v1',
                leaf_pixels=int(mask.sum()), review_flags=flags,
                warning='Whole-leaf mask generated automatically. Color extraction may miss brown tissue '
                        'or include background vegetation. The calculated area is an automatic estimate; '
                        'extraction has not been independently validated.')
