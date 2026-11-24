"""Visible leaf area only; unknown denominators never become a percentage."""
from dataclasses import asdict, dataclass
import io
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SeverityResult:
    severity_percent: float | None
    severity_valid: bool
    total_leaf_pixels: int
    diseased_leaf_pixels: int
    disease_pixels_outside_leaf: int
    reason: str

    def to_dict(self):
        return asdict(self)


def binary_mask(mask):
    array = np.asarray(mask)
    if array.ndim != 2 or array.size == 0:
        raise ValueError('Masks must be nonempty two-dimensional arrays.')
    if not np.isin(array, [0, 1, 255]).all():
        raise ValueError('Masks must use only 0/1 or 0/255 binary labels.')
    return array != 0


def calculate_severity(disease_mask, leaf_mask, *, reviewed=False, estimate_unreviewed=False):
    disease, leaf = binary_mask(disease_mask), binary_mask(leaf_mask)
    if disease.shape != leaf.shape:
        raise ValueError('Disease and leaf masks must have identical original-image dimensions.')
    total = int(leaf.sum())
    infected = int((disease & leaf).sum())
    outside = int((disease & ~leaf).sum())
    if total == 0:
        return SeverityResult(None, False, 0, 0, outside, 'No visible leaf pixels; denominator is zero.')
    if not reviewed:
        if estimate_unreviewed:
            return SeverityResult(100.0 * infected / total, False, total, infected, outside,
                                  'Automatic estimate; whole-leaf extraction has not been independently validated.')
        return SeverityResult(None, False, total, infected, outside,
                              'Whole-leaf mask requires image-specific review; severity unavailable.')
    return SeverityResult(100.0 * infected / total, True, total, infected, outside,
                          'Visible-area estimate using a user-reviewed leaf mask and draft-label-trained U-Net.')


def load_leaf_mask(data, image_size):
    """No silent resizing, RGB conversion, or label thresholding of external masks."""
    if not data or len(data) > 30 * 1024 * 1024:
        raise ValueError('Use a binary PNG leaf mask smaller than 30 MB.')
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format != 'PNG' or image.mode not in {'1', 'L', 'P'}:
                raise ValueError('Leaf mask must be a single-channel binary PNG.')
            if image.size != image_size:
                raise ValueError('Leaf mask must match the EXIF-corrected image dimensions exactly.')
            return binary_mask(np.asarray(image))
    except OSError as exc:
        raise ValueError('Cannot read the leaf-mask PNG.') from exc
