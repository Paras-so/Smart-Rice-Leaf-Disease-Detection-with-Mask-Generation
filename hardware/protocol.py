"""Version 1 newline-delimited JSON; reports cannot actuate hardware."""
import json
import math
from pathlib import Path
import re

CLASS_NAMES = tuple(json.loads((Path(__file__).resolve().parents[1] /
                               'results/resnet18_frozen/config.json').read_text())['classes'])
MESSAGE_TYPES = {'hello', 'prediction', 'status', 'arm_simulation', 'simulate', 'disarm', 'estop'}
MAX_LINE_BYTES = 2048


def number(value, low, high):
    return type(value) in (float, int) and math.isfinite(value) and low <= value <= high


def validate_message(message):
    if not isinstance(message, dict) or type(message.get('protocol_version')) is not int or message['protocol_version'] != 1:
        raise ValueError('Unsupported protocol version.')
    for field in ['message_id', 'session_id']:
        if not isinstance(message.get(field), str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', message[field]):
            raise ValueError(f'Invalid {field}.')
    if type(message.get('sequence')) is not int or not 1 <= message['sequence'] <= 2_000_000_000:
        raise ValueError('Invalid sequence number.')
    kind = message.get('type')
    if kind not in MESSAGE_TYPES:
        raise ValueError('Unsupported message type; physical actuation is not exposed.')
    if kind != 'hello' and (not isinstance(message.get('boot_id'), str) or not message['boot_id']):
        raise ValueError('A device boot ID from the handshake is required.')
    if kind == 'prediction':
        if message.get('disease_class') not in CLASS_NAMES:
            raise ValueError('Unknown disease class.')
        if not number(message.get('classification_confidence'), 0, 1):
            raise ValueError('Confidence must be finite and between zero and one.')
        if type(message.get('severity_valid')) is not bool:
            raise ValueError('severity_valid must be a boolean.')
        severity = message.get('severity_percent')
        if message['severity_valid']:
            if not number(severity, 0, 100):
                raise ValueError('Valid severity requires a finite percentage from zero to 100.')
        elif severity is not None:
            raise ValueError('Invalid severity must be represented as null.')
    return message


def encode_message(message):
    validate_message(message)
    line = (json.dumps(message, allow_nan=False, separators=(',', ':')) + '\n').encode('utf-8')
    if len(line) > MAX_LINE_BYTES:
        raise ValueError('Message is too long.')
    return line


def decode_object(line):
    if len(line) > MAX_LINE_BYTES or not line.endswith(b'\n'):
        raise ValueError('Expected a bounded newline-terminated JSON frame.')
    value = json.loads(line.decode('utf-8'), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object.')
    return value


def validate_ack(response, request, expected_boot=None):
    if (response.get('type') != 'ack' or response.get('protocol_version') != 1 or
        response.get('message_id') != request['message_id'] or
        response.get('session_id') != request['session_id'] or
        type(response.get('sequence')) is not int or response['sequence'] != request['sequence'] or
        type(response.get('ok')) is not bool or type(response.get('duplicate')) is not bool or
        not isinstance(response.get('boot_id'), str) or not response['boot_id'] or
        response.get('mode') != 'simulation' or response.get('pump_on') is not False or
        response.get('state') not in {'DISARMED', 'ARMED_SIMULATION', 'SIMULATED', 'ESTOP', 'FAULT'}):
        raise ValueError('Invalid or unmatched ESP32 acknowledgement.')
    if expected_boot and response['boot_id'] != expected_boot:
        raise ValueError('Device reset detected. Reconnect before sending further commands.')
    params = response.get('spray_parameters')
    if not isinstance(params, dict):
        raise ValueError('Missing spray telemetry.')
    for field in ['duration_ms', 'volume_ml']:
        if params.get(field) is not None and not number(params[field], 0, 1e9):
            raise ValueError('Invalid spray telemetry.')
    return response
