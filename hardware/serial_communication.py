"""One connection per owner. No implicit reconnect, retries, arming, or actuation."""
import threading
import time
import uuid
from hardware.protocol import encode_message, decode_object, validate_ack
from hardware.simulator import MockTransport


class CommunicationError(RuntimeError):
    pass


def available_ports():
    try:
        from serial.tools import list_ports
        return [(p.device, p.description) for p in list_ports.comports()]
    except ImportError:
        return []


class SerialClient:
    def __init__(self, timeout=2.0):
        self.timeout = timeout
        self.transport = None
        self.boot_id = None
        self.session_id = uuid.uuid4().hex
        self.sequence = 0
        self.lock = threading.RLock()
        self.last_ack = None

    @property
    def connected(self):
        return self.transport is not None and self.transport.is_open and self.boot_id is not None

    def connect(self, port, baudrate=115200, mock_config=None, transport=None):
        with self.lock:
            if self.transport is not None:
                raise CommunicationError('Disconnect the current device before opening another connection.')
            self.session_id = uuid.uuid4().hex; self.sequence = 0; self.boot_id = None
            try:
                if transport is not None:
                    self.transport = transport
                elif mock_config is not None:
                    self.transport = MockTransport(mock_config)
                else:
                    import serial
                    # Avoid asserting control lines before opening; board-specific USB reset may still occur.
                    self.transport = serial.Serial(port=None, baudrate=baudrate, timeout=self.timeout,
                                                   write_timeout=self.timeout)
                    self.transport.dtr = False; self.transport.rts = False
                    self.transport.port = port
                    self.transport.open()
                    time.sleep(1.5)  # ESP32 USB reset/startup; no output command is sent.
                    self.transport.reset_input_buffer()
                ack = self.request('hello')
                self.boot_id = ack['boot_id']
                return ack
            except Exception as exc:
                self.disconnect()
                raise CommunicationError(f'Connection failed: {exc}') from exc

    def request(self, kind, **fields):
        with self.lock:
            if self.transport is None or not self.transport.is_open:
                raise CommunicationError('Device is disconnected.')
            self.sequence += 1
            message = dict(protocol_version=1, type=kind, message_id=uuid.uuid4().hex,
                           session_id=self.session_id, sequence=self.sequence, **fields)
            if kind != 'hello': message['boot_id'] = self.boot_id
            encoded = encode_message(message)
            try:
                if self.transport.write(encoded) != len(encoded):
                    raise OSError('Incomplete serial write.')
                deadline = time.monotonic() + self.timeout
                while time.monotonic() < deadline:
                    line = self.transport.readline(2049)
                    if not line: raise TimeoutError('ESP32 acknowledgement timed out.')
                    response = decode_object(line)
                    ack = validate_ack(response, message, self.boot_id)
                    self.last_ack = ack
                    return ack
                raise TimeoutError('ESP32 acknowledgement timed out.')
            except Exception as exc:
                self.disconnect()
                raise CommunicationError(f'Communication failed; connection closed: {exc}') from exc

    def disconnect(self):
        with self.lock:
            transport, self.transport = self.transport, None
            self.boot_id = None
            if transport:
                try: transport.close()
                except Exception: pass

    def __del__(self):
        self.disconnect()
