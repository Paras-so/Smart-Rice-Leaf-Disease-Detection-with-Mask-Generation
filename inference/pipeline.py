"""Read-only trained model loading and inference. Never imports training scripts."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import threading
import numpy as np
import torch
from models.classification import build_classifier
from models.unet import UNet
from inference.preprocessing import classification_transform, letterbox, restore_probability

ROOT = Path(__file__).resolve().parents[1]
HEALTHY_CLASS = 'Healthy Rice Leaf'


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def settings():
    return json.loads((ROOT / 'config/app.json').read_text(encoding='utf-8'))


def choose_device(requested='auto'):
    if requested == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(requested)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Configured CUDA device is unavailable. Use auto or cpu.')
    return device


@dataclass
class LoadedModel:
    model: object
    config: dict
    checkpoint_hash: str
    device: object
    lock: object
    classes: tuple = ()


def load_classifier(path, device='auto'):
    path = Path(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    mapping = checkpoint['class_to_idx']
    classes = tuple(config['classes'])
    if len(classes) != 9 or mapping != {name: i for i, name in enumerate(classes)}:
        raise ValueError('Checkpoint class ordering is inconsistent.')
    if config['class_to_idx'] != mapping or HEALTHY_CLASS not in mapping:
        raise ValueError('Checkpoint configuration or Healthy label is inconsistent.')
    if config['hidden_layers'] != [512, 256, 128]:
        raise ValueError('Unexpected classification head.')
    if config['weights'] != 'ResNet18_Weights.IMAGENET1K_V1':
        raise ValueError('Unrecognized training preprocessing.')
    model = build_classifier(len(classes))
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    target = choose_device(device)
    model.to(target).eval().requires_grad_(False)
    return LoadedModel(model, config, file_sha256(path), target, threading.RLock(), classes)


def load_segmenter(path, device='auto'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if (checkpoint['architecture'], checkpoint['in_channels'], checkpoint['out_channels']) != ('UNet', 3, 1):
        raise ValueError('Expected the existing binary RGB U-Net.')
    if checkpoint['epoch'] < 1:
        raise ValueError('Expected trained weights.')
    config = checkpoint['config']
    model = UNet(base_channels=config['base_channels'])
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    target = choose_device(device)
    model.to(target).eval().requires_grad_(False)
    return LoadedModel(model, config, file_sha256(path), target, threading.RLock())


def classify(image, loaded):
    tensor = classification_transform()(image)
    with loaded.lock, torch.inference_mode():
        probabilities = loaded.model(tensor[None].to(loaded.device)).softmax(1)[0].cpu().numpy()
    if probabilities.shape != (9,) or not np.isfinite(probabilities).all():
        raise ValueError('Invalid classifier output.')
    index = int(probabilities.argmax())
    return dict(disease_class=loaded.classes[index], classification_confidence=float(probabilities[index]),
                probabilities={name: float(p) for name, p in zip(loaded.classes, probabilities)},
                checkpoint_sha256=loaded.checkpoint_hash, healthy=loaded.classes[index] == HEALTHY_CLASS)


def segment(image, loaded):
    tensor, geometry = letterbox(image, loaded.config)
    with loaded.lock, torch.inference_mode():
        output = loaded.model(tensor[None].to(loaded.device))
        probability = output.sigmoid()[0, 0].cpu().numpy()
    if not np.isfinite(probability).all():
        raise ValueError('Invalid segmentation output.')
    probability = restore_probability(probability, geometry, image.size)
    mask = probability >= loaded.config['threshold']
    overlay = np.asarray(image).copy()
    overlay[mask] = (overlay[mask] * 0.55 + np.array([255, 25, 38]) * 0.45).astype(np.uint8)
    return dict(mask=mask, probability=probability, overlay=overlay,
                diseased_pixels=int(mask.sum()), checkpoint_sha256=loaded.checkpoint_hash,
                threshold=loaded.config['threshold'])
