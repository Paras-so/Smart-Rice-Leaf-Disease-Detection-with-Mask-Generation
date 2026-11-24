"""Original evaluation transforms; classification and segmentation stay independent."""
import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights


def classification_transform():
    return ResNet18_Weights.IMAGENET1K_V1.transforms()


def letterbox(image, config):
    size = config['image_size']
    width, height = image.size
    scale = min(size / width, size / height)
    nw, nh = max(1, round(width * scale)), max(1, round(height * scale))
    left, top = (size - nw) // 2, (size - nh) // 2
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[top:top + nh, left:left + nw] = np.asarray(
        image.resize((nw, nh), Image.Resampling.BILINEAR))
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255
    mean = torch.tensor(config['normalization_mean'])[:, None, None]
    std = torch.tensor(config['normalization_std'])[:, None, None]
    return (tensor - mean) / std, (left, top, nw, nh)


def restore_probability(probability, geometry, original_size):
    left, top, width, height = geometry
    crop = probability[top:top + height, left:left + width]
    return np.asarray(Image.fromarray(crop).resize(original_size, Image.Resampling.BILINEAR))
