"""Read existing evaluation artifacts and verify provenance without recomputing scores."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from inference.pipeline import ROOT, file_sha256


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def classification_results():
    folder = ROOT / 'results/resnet18_frozen'
    config = read_json(folder / 'config.json')
    checkpoint = torch.load(folder / 'best_model.pt', map_location='cpu', weights_only=True)
    digest = file_sha256(folder / 'best_model.pt')
    with np.load(folder / 'test_outputs.npz', allow_pickle=False) as saved:
        linked = saved['model_sha256'].item() == digest
        expected_count = len(saved['y_true'])
        dimensions_ok = saved['probabilities'].shape == (expected_count, len(config['classes']))
    splits_ok = config['split_manifest_sha256'] == file_sha256(ROOT / 'results/provenance/split_manifest.csv')
    matrix = pd.read_csv(folder / 'confusion_matrix.csv', index_col=0)
    mapping_ok = list(matrix.index) == config['classes'] and list(matrix.columns) == config['classes']
    metrics = read_json(folder / 'test_metrics.json')
    checkpoint_ok = (checkpoint['config'] == config and checkpoint['class_to_idx'] == config['class_to_idx']
                     and metrics['best_epoch'] == checkpoint['epoch'])
    return dict(folder=folder, metrics=metrics,
                history=pd.read_csv(folder / 'history.csv'),
                report=pd.read_csv(folder / 'classification_report.csv', index_col=0),
                confusion_matrix=matrix, provenance_verified=bool(linked and splits_ok and dimensions_ok and mapping_ok and checkpoint_ok),
                provenance_note='Prediction cache links to checkpoint; saved split manifest hash and matrix labels checked. '
                                'Aggregate metrics are read as saved; they are not independently re-evaluated.',
                checkpoint_sha256=digest)


def segmentation_results():
    folder = ROOT / 'results/unet_disease_segmentation'
    metrics = read_json(folder / 'test_metrics.json')
    linked = metrics.get('checkpoint_sha256') == file_sha256(folder / 'best_model.pt')
    checkpoint = torch.load(folder / 'best_model.pt', map_location='cpu', weights_only=True)
    paths = {'split_manifest': ROOT / 'results/provenance/split_manifest.csv',
             'original_masks_manifest': ROOT / 'results/provenance/original_masks_manifest.csv',
             'augmented_masks_manifest': ROOT / 'results/provenance/augmented_masks_manifest.csv'}
    split_ok = all(checkpoint['manifest_sha256'].get(key) == file_sha256(path) for key, path in paths.items())
    return dict(folder=folder, metrics=metrics, history=pd.read_csv(folder / 'history.csv'),
                provenance_verified=linked and split_ok and metrics['checkpoint_epoch'] == checkpoint['epoch'])


def explainability_results():
    folder = ROOT / 'results/resnet18_frozen_gradcam'
    summary = read_json(folder / 'deletion_insertion_summary.json')
    status = read_json(folder / 'analysis_status.json')
    digest = file_sha256(ROOT / 'results/resnet18_frozen/best_model.pt')
    return dict(folder=folder, summary=summary,
                provenance_verified=summary['checkpoint_sha256'] == status['checkpoint_sha256'] == digest)
