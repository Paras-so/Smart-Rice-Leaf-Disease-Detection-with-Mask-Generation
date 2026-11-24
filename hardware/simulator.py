"""Deterministic ESP32 contract simulator. Never imports GPIO or activates a pump."""
import json
import time
import uuid
from hardware.protocol import validate_message, number


class DeviceSimulator:
    def __init__(self, config, clock=time.monotonic):
        self.config = dict(config)
        self.clock = clock
        self.boot_id = uuid.uuid4().hex
        self.session_id = None
        self.last_sequence = 0
        self.last_request = None
        self.last_ack = None
        self.last_seen = clock()
        self.state = 'DISARMED'
        self.prediction = None
        self.simulated_commands = 0

    def expire(self):
        if self.clock() - self.last_seen > self.config.get('communication_timeout_seconds', 5):
            if self.state != 'ESTOP': self.state = 'DISARMED'
            self.prediction = None

    def plan(self):
        c = self.config
        if not c.get('calibration_validated') or not c.get('application_limits_validated'):
            return None, 'Calibration and application limits are not validated.'
        for field in ['flow_ml_per_second', 'target_volume_ml', 'available_volume_ml', 'tank_capacity_ml', 'maximum_duration_ms']:
            if not number(c.get(field), 1e-9, 1e9):
                return None, f'Missing or invalid hardware setting: {field}'
        duration = c['target_volume_ml'] / c['flow_ml_per_second'] * 1000
        if (duration > c['maximum_duration_ms'] or c['target_volume_ml'] > c['available_volume_ml'] or
                c['available_volume_ml'] > c['tank_capacity_ml']):
            return None, 'Requested volume or duration exceeds configured limits.'
        return {'duration_ms': duration, 'volume_ml': c['target_volume_ml']}, ''

    def handle(self, request):
        validate_message(request)
        self.expire()
        kind = request['type']
        response = dict(protocol_version=1, type='ack', message_id=request['message_id'],
                        session_id=request['session_id'], sequence=request['sequence'],
                        boot_id=self.boot_id, ok=True, duplicate=False, mode='simulation', pump_on=False)
        reason = ''
        if kind != 'hello' and (request['boot_id'] != self.boot_id or request['session_id'] != self.session_id):
            response['ok'] = False; reason = 'Stale boot or session ID; reconnect.'
        elif self.last_request == request:
            # Replay the acknowledgement without executing the command again.
            return {**self.last_ack, 'duplicate': True, 'state': self.state}
        elif kind != 'hello' and request['sequence'] <= self.last_sequence:
            response['ok'] = False; reason = 'Old or reused sequence rejected.'
        elif kind == 'hello' and request['session_id'] == self.session_id:
            response['ok'] = False; reason = 'Repeated session handshake rejected.'
        else:
            self.last_seen = self.clock()
            self.last_sequence = request['sequence']
            if kind == 'hello':
                self.session_id = request['session_id']; self.prediction = None
                # A latched emergency stop needs a device reset, not a new host session.
                if self.state != 'ESTOP': self.state = 'DISARMED'
            elif kind == 'estop':
                self.state = 'ESTOP'; self.prediction = None
            elif self.state == 'ESTOP' and kind not in {'status', 'disarm'}:
                response['ok'] = False; reason = 'Emergency stop latched; reset device.'
            elif kind == 'disarm':
                if self.state != 'ESTOP': self.state = 'DISARMED'
                self.prediction = None
            elif kind == 'prediction':
                self.prediction = request.copy(); self.state = 'DISARMED'
            elif kind in {'arm_simulation', 'simulate'}:
                p = self.prediction
                plan, reason = self.plan()
                if not p or p['disease_class'] == 'Healthy Rice Leaf' or not p['severity_valid'] or p['severity_percent'] <= 0:
                    response['ok'] = False; reason = 'Healthy, absent, empty, or invalid severity prediction.'
                elif plan is None:
                    response['ok'] = False
                elif kind == 'arm_simulation':
                    self.state = 'ARMED_SIMULATION'
                elif self.state != 'ARMED_SIMULATION':
                    response['ok'] = False; reason = 'Explicit simulation arm required.'
                else:
                    self.state = 'SIMULATED'; self.simulated_commands += 1
                    self.config['available_volume_ml'] -= plan['volume_ml']
                    response['spray_parameters'] = plan
                    self.prediction = None  # A fresh report and arm are required for another cycle.
        response.update(state=self.state, reason=reason)
        plan, blocked = self.plan()
        response.setdefault('spray_parameters', plan or {'duration_ms': None, 'volume_ml': None})
        response['configuration_status'] = blocked or 'Configured simulation only.'
        if response['ok']:
            self.last_request, self.last_ack = request.copy(), response.copy()
        return response


class MockTransport:
    def __init__(self, config):
        self.device = DeviceSimulator(config)
        self.is_open = True
        self.reply = b''

    def write(self, data):
        if not self.is_open: raise OSError('Mock device disconnected.')
        self.reply = (json.dumps(self.device.handle(json.loads(data))) + '\n').encode()
        return len(data)

    def readline(self, size=2048):
        if not self.is_open: raise OSError('Mock device disconnected.')
        reply, self.reply = self.reply, b''
        return reply[:size]

    def close(self):
        self.is_open = False
