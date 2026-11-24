/* ESP32 / ArduinoJson 7. Simulation only: no command energizes a relay.
   Newline-delimited protocol v1; do not print debug text onto this serial port. */
#include <Arduino.h>
#include <ArduinoJson.h>
#include <esp_system.h>
#include <math.h>
#include "HardwareConfig.h"
#include "SafetyController.h"
using namespace HardwareConfig;

String bootId, sessionId, lastId, lastFrame, lastReply;
uint32_t lastSequence = 0, lastCommunication = 0;
char frame[2049];
size_t used = 0;
bool overflow = false;
enum SimulationState { DISARMED, ARMED_SIMULATION, SIMULATED, ESTOP, FAULT };
SimulationState state = DISARMED;
SafetyController physicalController; // Stage B is not armed by the simulation protocol.
bool predictionPresent = false, severityValid = false;
float severityPercent = 0, confidence = 0, remainingVolume = AVAILABLE_VOLUME_ML;
String diseaseClass;
const char* classes[] = {"Bacterial Leaf Blight", "Brown Spot", "Healthy Rice Leaf", "Leaf Blast",
  "Leaf scald", "Narrow Brown Leaf Spot", "Rice Hispa", "Rice Turgro", "Sheath Blight"};

const char* stateName() {
  switch (state) {
    case ARMED_SIMULATION: return "ARMED_SIMULATION";
    case SIMULATED: return "SIMULATED";
    case ESTOP: return "ESTOP";
    case FAULT: return "FAULT";
    default: return "DISARMED";
  }
}

void relayOff() {
  if (RELAY_PIN >= 0 && RELAY_POLARITY_VALIDATED)
    digitalWrite(RELAY_PIN, RELAY_ACTIVE_LOW ? HIGH : LOW);
}

bool identifier(JsonVariantConst value) {
  if (!value.is<const char*>()) return false;
  const char* s = value.as<const char*>();
  size_t n = strlen(s);
  if (!n || n > 64) return false;
  for (size_t i = 0; i < n; i++)
    if (!isalnum(static_cast<unsigned char>(s[i])) && s[i] != '_' && s[i] != '-') return false;
  return true;
}

bool numeric(JsonVariantConst v, double low, double high) {
  if (v.is<bool>() || !(v.is<double>() || v.is<long>() || v.is<unsigned long>())) return false;
  double x = v.as<double>();
  return isfinite(x) && x >= low && x <= high;
}

bool knownClass(const String& value) {
  for (const char* c : classes) if (value == c) return true;
  return false;
}

bool calculatePlan(float& durationMs, String& reason) {
  if (!CALIBRATION_VALIDATED || !APPLICATION_LIMITS_VALIDATED) {
    reason = "Calibration and application limits are not validated."; return false;
  }
  if (!isfinite(FLOW_ML_PER_SECOND) || !isfinite(TARGET_VOLUME_ML) || !isfinite(remainingVolume) ||
      !isfinite(TANK_CAPACITY_ML) || FLOW_ML_PER_SECOND <= 0 || TARGET_VOLUME_ML <= 0 ||
      remainingVolume <= 0 || TANK_CAPACITY_ML <= 0 || MAXIMUM_DURATION_MS == 0) {
    reason = "Missing or invalid measured hardware settings."; return false;
  }
  durationMs = TARGET_VOLUME_ML / FLOW_ML_PER_SECOND * 1000.0f;
  if (!isfinite(durationMs) || durationMs > MAXIMUM_DURATION_MS ||
      TARGET_VOLUME_ML > remainingVolume || remainingVolume > TANK_CAPACITY_ML) {
    reason = "Requested volume or duration exceeds configured limits."; return false;
  }
  return true;
}

void emit(JsonDocument& doc) { serializeJson(doc, Serial); Serial.println(); }

void processFrame(const String& raw) {
  JsonDocument input;
  auto error = deserializeJson(input, raw);
  if (error || !input.is<JsonObject>()) { relayOff(); state = FAULT; predictionPresent = false; return; }
  if (!input["protocol_version"].is<int>() || input["protocol_version"].as<int>() != 1 ||
      !identifier(input["message_id"]) || !identifier(input["session_id"]) ||
      input["sequence"].is<bool>() || !input["sequence"].is<uint32_t>() ||
      input["sequence"].as<uint32_t>() < 1 || input["sequence"].as<uint32_t>() > 2000000000UL ||
      !input["type"].is<const char*>()) { relayOff(); state = FAULT; predictionPresent = false; return; }
  String kind = input["type"].as<String>();
  String id = input["message_id"].as<String>();
  String session = input["session_id"].as<String>();
  uint32_t sequence = input["sequence"].as<uint32_t>();
  bool allowed = kind == "hello" || kind == "prediction" || kind == "status" || kind == "arm_simulation" ||
                 kind == "simulate" || kind == "disarm" || kind == "estop";
  JsonDocument reply;
  reply["protocol_version"] = 1; reply["type"] = "ack";
  reply["message_id"] = id; reply["session_id"] = session; reply["sequence"] = sequence;
  reply["boot_id"] = bootId; reply["mode"] = "simulation";
  reply["pump_on"] = false; reply["duplicate"] = false;
  bool ok = true;
  String reason;
  if (!allowed) { ok = false; reason = "Unsupported command; physical actuation is not exposed."; }
  else if (kind != "hello" && (input["boot_id"].as<String>() != bootId || session != sessionId)) {
    ok = false; reason = "Stale boot or session ID; reconnect.";
  } else if (raw == lastFrame && id == lastId) {
    JsonDocument duplicate; deserializeJson(duplicate, lastReply);
    duplicate["duplicate"] = true; duplicate["state"] = stateName(); emit(duplicate); return;
  } else if (kind != "hello" && sequence <= lastSequence) {
    ok = false; reason = "Old or reused sequence rejected.";
  } else if (kind == "hello" && session == sessionId) {
    ok = false; reason = "Repeated session handshake rejected.";
  } else if (kind == "prediction" &&
      (!input["disease_class"].is<const char*>() || !knownClass(input["disease_class"].as<String>()) ||
       !numeric(input["classification_confidence"], 0, 1) || !input["severity_valid"].is<bool>() ||
       (input["severity_valid"].as<bool>() ? !numeric(input["severity_percent"], 0, 100) :
        (!input.containsKey("severity_percent") || !input["severity_percent"].isNull())))) {
    ok = false; reason = "Invalid prediction schema.";
  } else {
    lastSequence = sequence; lastCommunication = millis();
    if (kind == "hello") {
      sessionId = session; predictionPresent = false;
      if (state != ESTOP && state != FAULT) state = DISARMED;
    } else if (kind == "estop") {
      state = ESTOP; predictionPresent = false; physicalController.emergencyStop();
    } else if ((state == ESTOP || state == FAULT) && kind != "status" && kind != "disarm") {
      ok = false; reason = "Emergency stop or fault latched; reset device.";
    } else if (kind == "disarm") {
      if (state != ESTOP && state != FAULT) state = DISARMED;
      predictionPresent = false;
    } else if (kind == "prediction") {
      diseaseClass = input["disease_class"].as<String>();
      confidence = input["classification_confidence"].as<float>();
      severityValid = input["severity_valid"].as<bool>();
      severityPercent = severityValid ? input["severity_percent"].as<float>() : 0;
      predictionPresent = true; state = DISARMED;
    } else if (kind == "arm_simulation" || kind == "simulate") {
      float durationMs = 0;
      if (!predictionPresent || diseaseClass == "Healthy Rice Leaf" || !severityValid || severityPercent <= 0) {
        ok = false; reason = "Healthy, absent, empty, or invalid severity prediction.";
      } else if (!calculatePlan(durationMs, reason)) ok = false;
      else if (kind == "arm_simulation") state = ARMED_SIMULATION;
      else if (state != ARMED_SIMULATION) { ok = false; reason = "Explicit simulation arm required."; }
      else {
        state = SIMULATED; predictionPresent = false;
        reply["spray_parameters"]["duration_ms"] = durationMs;
        reply["spray_parameters"]["volume_ml"] = TARGET_VOLUME_ML;
        remainingVolume -= TARGET_VOLUME_ML;
      }
    }
  }
  relayOff(); // Unconditional: this build has no relay-ON path.
  float durationMs = 0; String blocked;
  bool configured = calculatePlan(durationMs, blocked);
  if (!reply["spray_parameters"].is<JsonObject>()) {
    if (configured) {
      reply["spray_parameters"]["duration_ms"] = durationMs;
      reply["spray_parameters"]["volume_ml"] = TARGET_VOLUME_ML;
    } else {
      reply["spray_parameters"]["duration_ms"] = nullptr;
      reply["spray_parameters"]["volume_ml"] = nullptr;
    }
  }
  reply["configuration_status"] = configured ? "Configured simulation only." : blocked;
  reply["ok"] = ok; reply["reason"] = reason; reply["state"] = stateName();
  if (ok) { lastId = id; lastFrame = raw; lastReply = ""; serializeJson(reply, lastReply); }
  emit(reply);
}

void setup() {
  // Electrical pull resistors / relay wiring must independently enforce OFF
  // during boot before firmware runs. Do not connect a pump for Stage A tests.
  if (RELAY_PIN >= 0 && RELAY_POLARITY_VALIDATED) {
    relayOff(); pinMode(RELAY_PIN, OUTPUT); relayOff();
  }
  if (EMERGENCY_STOP_PIN >= 0) pinMode(EMERGENCY_STOP_PIN, INPUT_PULLUP);
  if (FAULT_INPUT_PIN >= 0) pinMode(FAULT_INPUT_PIN, INPUT_PULLUP);
  Serial.begin(BAUD_RATE);
  char id[17]; snprintf(id, sizeof(id), "%08lx%08lx", (unsigned long)esp_random(), (unsigned long)esp_random());
  bootId = id; lastCommunication = millis();
}

void loop() {
  relayOff();
  bool emergency = EMERGENCY_STOP_PIN >= 0 && digitalRead(EMERGENCY_STOP_PIN) == LOW;
  bool fault = FAULT_INPUT_PIN >= 0 && digitalRead(FAULT_INPUT_PIN) == LOW;
  if (emergency) { state = ESTOP; predictionPresent = false; }
  if (fault && state != ESTOP) { state = FAULT; predictionPresent = false; }
  physicalController.tick(millis(), COMMUNICATION_TIMEOUT_MS, emergency, fault);
  if (uint32_t(millis() - lastCommunication) >= COMMUNICATION_TIMEOUT_MS) {
    if (state != ESTOP && state != FAULT) state = DISARMED;
    predictionPresent = false;
  }
  // Bound work per iteration so a continuous stream cannot starve safety checks.
  for (unsigned count = 0; Serial.available() && count < 128; count++) {
    char c = Serial.read();
    if (c == '\n') {
      if (overflow) { state = FAULT; predictionPresent = false; }
      else if (used) { frame[used] = '\0'; processFrame(String(frame)); }
      used = 0; overflow = false;
    } else if (c != '\r') {
      if (used < sizeof(frame) - 1 && !overflow) frame[used++] = c;
      else overflow = true;
    }
  }
}
