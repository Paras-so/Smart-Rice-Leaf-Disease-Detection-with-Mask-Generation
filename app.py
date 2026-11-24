"""Rice leaf demonstration: trained inference, reviewed severity, simulation-only control."""
import hashlib
import importlib
import inspect
import json
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch
from inference.pipeline import ROOT, settings, load_classifier, load_segmenter, classify, segment
from inference.leaf_extraction import propose_leaf_mask
from inference import severity_calculation as severity_module
from hardware.serial_communication import SerialClient, CommunicationError, available_ports
from utils.image_processing import decode_image, png_bytes, image_preview
from utils.pesticides import PESTICIDES
from utils.metrics_loader import classification_results, segmentation_results, explainability_results, read_json

# A Streamlit script rerun can keep an older imported module in memory after
# the app and helper are edited together. Refresh this stateless helper once
# when its loaded API predates automatic estimates.
if 'estimate_unreviewed' not in inspect.signature(severity_module.calculate_severity).parameters:
    importlib.invalidate_caches()
    severity_module = importlib.reload(severity_module)
calculate_severity = severity_module.calculate_severity

st.set_page_config(page_title='Rice Leaf AI', page_icon='🌾', layout='wide')
torch.set_num_threads(4)
CONFIG = settings()
CLASSIFIER_PATH = ROOT / CONFIG['classifier']


@st.cache_resource(show_spinner='Loading trained classifier…')
def cached_classifier(path, modified):
    return load_classifier(path, CONFIG['device'])


@st.cache_resource(show_spinner='Loading trained segmenter…')
def cached_segmenter(path, modified):
    return load_segmenter(path, CONFIG['device'])


@st.cache_resource(scope='session', on_release=lambda client: client.disconnect())
def device_client():
    return SerialClient(CONFIG['serial']['timeout'])


@st.cache_data
def cached_evaluation(kind, fingerprint):
    if kind == 'classification': return classification_results()
    if kind == 'segmentation': return segmentation_results()
    return explainability_results()


def go_segmentation():
    st.session_state.navigation = 'Segmentation & Severity'


def show_preview(image, caption, *, container=None, mask=False):
    preview = image_preview(image, mask=mask)
    target = container if container is not None else st
    target.image(preview, caption=caption, width=preview.width)


def reset_image():
    for key in ['image', 'image_id', 'classification', 'classification_key', 'segmentation',
                'segmentation_key', 'proposal', 'proposal_key', 'leaf_review', 'leaf_upload', 'leaf_mask_digest', 'sent_report_key']:
        st.session_state.pop(key, None)


def photo_upload_changed():
    # A deliberate removal must clear old predictions. Merely navigating away
    # from this page does not invoke this callback and keeps the shared image.
    if st.session_state.get('photo_upload') is None:
        reset_image()


def current_classifier():
    return cached_classifier(str(CLASSIFIER_PATH), CLASSIFIER_PATH.stat().st_mtime_ns)


def get_classification(image):
    loaded = current_classifier()
    key = (st.session_state.image_id, loaded.checkpoint_hash)
    if st.session_state.get('classification_key') != key:
        st.session_state.classification = classify(image, loaded)
        st.session_state.classification_key = key
    return st.session_state.classification


def classification_page():
    st.title('Rice Leaf Disease Classification')
    st.write('Upload a rice leaf photograph.')
    uploaded = st.file_uploader('Upload JPEG, PNG, HEIC, or HEIF', type=['jpg', 'jpeg', 'png', 'heic', 'heif'],
                               key='photo_upload', max_upload_size=30, on_change=photo_upload_changed)
    if uploaded:
        data = uploaded.getvalue()
        image_id = hashlib.sha256(data).hexdigest()
        if image_id != st.session_state.get('image_id'):
            try:
                image = decode_image(data)
            except ValueError as exc:
                reset_image(); st.error(str(exc)); return
            reset_image()
            st.session_state.image, st.session_state.image_id = image, image_id
    image = st.session_state.get('image')
    if image is None:
        st.info('Choose a photo to begin.')
        return
    left, right = st.columns([1.2, 1])
    show_preview(image, f'Original: {image.width} × {image.height}', container=left)
    with right:
        if st.button('Classify image', type='primary'):
            with st.spinner('Running the trained ResNet-18…'):
                get_classification(image)
        result = st.session_state.get('classification')
        if result and st.session_state.get('classification_key', (None, None))[1] == current_classifier().checkpoint_hash:
            st.metric('Predicted disease', result['disease_class'])
            st.write('Required pesticide: ' + PESTICIDES.get(result['disease_class'], 'No recommendation available'))
            st.metric('Classification confidence', f"{result['classification_confidence']:.2%}")
            if result['healthy']:
                st.success('Healthy prediction: spraying is blocked. This prediction does not prove absence of disease.')
            probabilities = pd.DataFrame({'Probability': result['probabilities']})
            st.bar_chart(probabilities)
            st.button('Continue to segmentation', on_click=go_segmentation)


def segmentation_page():
    st.title('Disease Segmentation and Severity Estimation')
    image = st.session_state.get('image')
    if image is None:
        st.info('Upload a photo on Disease Classification first.'); return
    classification = get_classification(image)
    st.caption(f"Classifier: {classification['disease_class']} · confidence {classification['classification_confidence']:.2%}")
    st.caption('Photo → automatic whole-leaf mask + U-Net disease segmentation → affected leaf area')
    st.caption('Affected area is estimated automatically using the generated leaf mask and U-Net disease mask.')
    if st.button('Segment diseased regions', type='primary'):
        if st.session_state.get('proposal_key') != st.session_state.image_id:
            with st.spinner('Finding the complete visible leaf region…'):
                st.session_state.proposal = propose_leaf_mask(image)
                st.session_state.proposal_key = st.session_state.image_id
        path = ROOT / CONFIG['segmenter']
        loaded = cached_segmenter(str(path), path.stat().st_mtime_ns)
        key = (st.session_state.image_id, loaded.checkpoint_hash)
        if st.session_state.get('segmentation_key') != key:
            with st.spinner('Running the trained U-Net…'):
                st.session_state.segmentation = segment(image, loaded)
                st.session_state.segmentation_key = key
    result = st.session_state.get('segmentation')
    if result is None:
        show_preview(image, 'Original photo')
        st.info('Run segmentation to inspect the disease mask and prepare a severity report.'); return
    columns = st.columns(3)
    show_preview(image, 'Original image', container=columns[0])
    show_preview(result['mask'], 'U-Net disease mask', container=columns[1], mask=True)
    show_preview(result['overlay'], 'Predicted symptoms in red', container=columns[2])
    st.success(f"Segmentation completed · {result['diseased_pixels']:,} predicted disease pixels · threshold {result['threshold']}")
    if st.session_state.get('proposal_key') != st.session_state.image_id:
        st.session_state.proposal = propose_leaf_mask(image)
        st.session_state.proposal_key = st.session_state.image_id
    proposal = st.session_state.proposal
    leaf = proposal['mask']
    source = 'automatic_color_extraction'
    st.session_state.leaf_mask_digest = hashlib.sha256(leaf.tobytes() + source.encode()).hexdigest()
    severity = calculate_severity(result['mask'], leaf, estimate_unreviewed=True)
    values = severity.to_dict()
    values.update(leaf_mask_source=source, leaf_extraction_independently_validated=False)
    st.subheader('Affected area')
    a, b, c = st.columns(3)
    a.metric('Total leaf pixels', f'{severity.total_leaf_pixels:,}')
    b.metric('Diseased pixels inside leaves', f'{severity.diseased_leaf_pixels:,}')
    c.metric('Affected visible leaf area', f'{severity.severity_percent:.2f}%' if severity.severity_percent is not None else 'Unavailable')
    if severity.severity_percent is not None:
        st.write(f'Affected area = ({severity.diseased_leaf_pixels:,} / {severity.total_leaf_pixels:,}) x 100 = {severity.severity_percent:.2f}%')
        st.caption(severity.reason)
    else:
        st.warning(severity.reason)
    if proposal.get('review_flags'):
        st.caption('Extraction notes: ' + ' '.join(proposal['review_flags']))
    st.caption(f'{severity.disease_pixels_outside_leaf:,} predicted disease pixels fall outside the supplied leaf mask and are excluded.')
    if classification['healthy']:
        if result['diseased_pixels'] > 0:
            st.warning('Model disagreement: the classifier predicts Healthy, but U-Net marked symptom pixels. '
                       'Review the image and mask; spraying remains blocked.')
        st.info('Healthy-class policy: keep the raw segmentation visible for review; block all spray simulation requests. '
                'A nonempty disease mask is a disagreement requiring review, not grounds to override Healthy automatically.')
    st.download_button('Download disease mask', png_bytes(result['mask']), 'disease_mask.png', 'image/png')
    st.download_button('Download area calculation', json.dumps(values, indent=2), 'severity.json', 'application/json')
    report = dict(disease_class=classification['disease_class'],
                  classification_confidence=classification['classification_confidence'],
                  severity_percent=severity.severity_percent if severity.severity_valid else None,
                  severity_valid=severity.severity_valid)
    serial_panel(report, classification['healthy'], st.session_state.image_id,
                 classification['checkpoint_sha256'], result['checkpoint_sha256'])


def serial_panel(report, healthy, *identity):
    st.divider()
    st.subheader('ESP32 connection and simulation')
    st.caption('Prediction reports cannot start the pump. Physical spraying is disabled in this application and firmware.')
    client = device_client()
    mode = st.radio('Connection mode', ['Simulator', 'USB ESP32'], horizontal=True, disabled=client.connected)
    port = ''
    if mode == 'USB ESP32':
        ports = available_ports()
        st.caption('Detected ports: ' + (', '.join(f'{p}: {d}' for p, d in ports) or 'none (pyserial required)'))
        port = st.text_input('Serial port', value=CONFIG['serial']['port'], placeholder='COM3')
    st.write('Status: ' + ('Connected' if client.connected else 'Disconnected'))
    cols = st.columns(3)
    if cols[0].button('Connect', disabled=client.connected):
        try:
            hardware = read_json(ROOT / 'config/hardware.json')
            ack = client.connect(port, CONFIG['serial']['baudrate'], mock_config=hardware if mode == 'Simulator' else None)
            st.session_state.device_ack = ack
            st.session_state.pop('sent_report_key', None)
            st.rerun()
        except CommunicationError as exc: st.error(str(exc))
    if cols[1].button('Disconnect', disabled=not client.connected):
        client.disconnect(); st.session_state.pop('sent_report_key', None); st.rerun()
    if cols[2].button('Read device status', disabled=not client.connected):
        send_device(client, 'status')
    st.write('Prediction payload (IDs and boot/session identifiers are added on transmission)')
    st.json({'protocol_version': 1, 'type': 'prediction', **report})
    report_key = hashlib.sha256(json.dumps([report, identity, st.session_state.get('leaf_mask_digest'),
                                          client.session_id], sort_keys=True).encode()).hexdigest()
    if st.button('Send prediction report', disabled=not client.connected):
        ack = send_device(client, 'prediction', **report)
        if ack and ack['ok']: st.session_state.sent_report_key = report_key
    eligible = (client.connected and report['severity_valid'] and not healthy and
                (report['severity_percent'] or 0) > 0 and st.session_state.get('sent_report_key') == report_key)
    cols = st.columns(3)
    if cols[0].button('Arm simulation', disabled=not eligible):
        send_device(client, 'arm_simulation')
    ack = st.session_state.get('device_ack', {})
    if cols[1].button('Run one simulated cycle', disabled=not eligible or ack.get('state') != 'ARMED_SIMULATION'):
        send_device(client, 'simulate')
        st.session_state.pop('sent_report_key', None)
    if cols[2].button('Emergency stop', disabled=not client.connected):
        send_device(client, 'estop'); st.session_state.pop('sent_report_key', None)
    if st.session_state.get('device_ack'):
        st.write('Last device acknowledgement (use Read device status to refresh)')
        st.json(st.session_state.device_ack)


def send_device(client, kind, **fields):
    try:
        ack = client.request(kind, **fields)
        st.session_state.device_ack = ack
        if not ack['ok']: st.warning(ack.get('reason', 'Device rejected the request.'))
        return ack
    except (CommunicationError, ValueError) as exc:
        st.error(str(exc)); return None


def evaluation_page():
    st.title('AI Model Evaluation and Performance')
    st.caption('Saved results only. This page does not run inference, training, or dataset evaluation.')
    first, second, third = st.tabs(['Classification Performance', 'Segmentation Performance', 'Explainability'])
    with first:
        folder = ROOT / 'results/resnet18_frozen'
        result = cached_evaluation('classification', (folder / 'best_model.pt').stat().st_mtime_ns)
        show_provenance(result)
        metrics = result['metrics']
        cols = st.columns(4)
        for col, (label, key) in zip(cols, [('Accuracy', 'accuracy'), ('Macro precision', 'macro_precision'),
                                         ('Macro recall', 'macro_recall'), ('Macro F1', 'macro_f1')]):
            col.metric(label, f'{metrics[key]:.4f}')
        st.caption('Accuracy: fraction classified correctly. Precision: reliability of positive class predictions. '
                   'Recall: fraction of class examples recovered. F1: precision/recall harmonic mean. Macro: equal class weighting.')
        st.caption(f'Source: results/resnet18_frozen/test_metrics.json; selected epoch {metrics["best_epoch"]}.')
        st.image(str(folder / 'learning_curves.png'), caption='Saved training and validation curves', width='stretch')
        st.image(str(folder / 'confusion_matrices.png'), caption='Saved confusion matrices', width='stretch')
        st.dataframe(result['report'])
        st.caption(f'Sources: results/resnet18_frozen/classification_report.csv and confusion_matrix.csv')
        with st.expander('Saved history and aggregate metrics'):
            st.dataframe(result['history']); st.json(metrics)
    with second:
        folder = ROOT / 'results/unet_disease_segmentation'
        result = cached_evaluation('segmentation', (folder / 'best_model.pt').stat().st_mtime_ns)
        show_provenance(result)
        metrics = result['metrics']
        cols = st.columns(4)
        for col, key in zip(cols, ['dice', 'iou', 'precision', 'recall']): col.metric(key.title(), f'{metrics[key]:.4f}')
        st.caption('Dice and IoU measure predicted/target region overlap. Pixel precision and recall measure false-positive '
                   'and missed-target behavior. These targets are automatic drafts, not expert reference masks.')
        st.caption('Source: results/unet_disease_segmentation/test_metrics.json')
        st.image(str(folder / 'training_curves.png'), caption='Saved segmentation learning curves', width='stretch')
        sample = folder / 'predictions/sample_test_pred_mask.png'
        if sample.exists(): st.image(str(sample), caption='Saved sample prediction in model-input coordinates (not native photo coordinates)', width=256)
        st.info('No saved paired U-Net prediction/ground-truth comparison panel was found. Annotation contact sheets are draft labels, not model comparisons.')
        with st.expander('All saved segmentation metrics and history'):
            st.json(metrics); st.dataframe(result['history'])
    with third:
        folder = ROOT / 'results/resnet18_frozen_gradcam'
        result = cached_evaluation('explainability', (folder / 'analysis_status.json').stat().st_mtime_ns)
        show_provenance(result)
        st.info(''
                'Grad-CAM is an attention explanation, not the disease segmentation mask.')
        summary = result['summary']
        a, b = st.columns(2)
        a.metric('Deletion AUC', f'{summary["deletion_auc"]:.4f}')
        b.metric('Insertion AUC', f'{summary["insertion_auc"]:.4f}')
        st.caption('Deletion: remove salient pixels; lower AUC is better under its baseline. Insertion: restore salient pixels; '
                   'higher AUC is better under its baseline. The baselines differ; the two AUC values are not directly comparable.')
        st.image(str(folder / 'deletion_insertion_curves.png'), width='stretch')
        st.image(str(folder / 'gradcam_examples_1.png'), caption='Saved frozen-model Grad-CAM examples', width='stretch')
        with st.expander('Additional saved example panels'):
            for name in ['gradcam_examples_2.png', 'gradcam_examples_3.png']: st.image(str(folder / name), width='stretch')
        st.caption('Sources: results/resnet18_frozen_gradcam/deletion_insertion_summary.json, deletion_insertion_curves.png and gradcam_examples_*.png')


def show_provenance(result):
    if result['provenance_verified']: st.success('Saved artifact provenance checks passed.')
    else: st.error('Saved artifact provenance mismatch. Treat these results as unverified.')
    if result.get('provenance_note'): st.caption(result['provenance_note'])


st.markdown('### Rice Leaf AI')
pages = ['Disease Classification', 'Segmentation & Severity', 'Model Evaluation']
page_tabs = st.tabs(pages, key='navigation', on_change='rerun')
try:
    # Render only the selected slide so hidden tabs do not run inference or
    # create duplicate upload/connection controls. The navigation callback can
    # still select the segmentation tab after classification.
    for tab, render in zip(page_tabs, [classification_page, segmentation_page, evaluation_page]):
        if tab.open:
            with tab:
                render()
except (ValueError, RuntimeError, OSError, KeyError) as exc:
    st.error(f'Unable to complete this operation: {exc}')
    st.info('Check model paths and dependencies in README.md. Existing checkpoints and results are never overwritten.')
