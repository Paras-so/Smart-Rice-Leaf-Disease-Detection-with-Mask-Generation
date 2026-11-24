import io
import warnings
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 30 * 1024 * 1024
MAX_PIXELS = 25_000_000


def image_preview(image, *, mask=False, max_width=640, max_height=400):
    """Bound display size without cropping or changing the inference image."""
    if isinstance(image, Image.Image):
        preview = image.copy()
    else:
        array = np.asarray(image)
        if array.dtype == bool:
            array = array.astype(np.uint8) * 255
        preview = Image.fromarray(array)
    preview.thumbnail((max_width, max_height),
                      Image.Resampling.NEAREST if mask else Image.Resampling.LANCZOS)
    return preview


def decode_image(data):
    if not data or len(data) > MAX_BYTES:
        raise ValueError('Upload an image smaller than 30 MB.')
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass  # JPEG and PNG continue to work; HEIF fails with a clear message below.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in {'JPEG', 'PNG', 'HEIF', 'HEIC'}:
                    raise ValueError('Use JPEG, PNG, or supported HEIC/HEIF files.')
                if source.width * source.height > MAX_PIXELS:
                    raise ValueError('Image exceeds the 25-megapixel limit.')
                source.load()
                return ImageOps.exif_transpose(source).convert('RGB')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError('Cannot decode this image. For HEIC/HEIF, install pillow-heif or export JPEG.') from exc


def png_bytes(array):
    array = np.asarray(array)
    if array.dtype == bool:
        array = array.astype(np.uint8) * 255
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format='PNG')
    return stream.getvalue()
