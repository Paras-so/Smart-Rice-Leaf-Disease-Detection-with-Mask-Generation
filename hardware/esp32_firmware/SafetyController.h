#pragma once
#include <stdint.h>

// Hardware-independent Stage B scaffold. No serial command can activate it in
// the supplied simulation firmware. The output backend must still fail closed.
class SafetyController {
 public:
  enum State { DISARMED, ARMED, SPRAYING, ESTOP, FAULT };
  State state = DISARMED;
  uint32_t started = 0, duration = 0, lastCommunication = 0;

  bool arm(uint32_t now, bool validated, bool interlocksOK) {
    if (!validated || !interlocksOK || state != DISARMED) return false;
    state = ARMED; lastCommunication = now; return true;
  }
  bool start(uint32_t now, uint32_t requested, uint32_t maximum, bool predictionSafe) {
    if (state != ARMED || !predictionSafe || requested == 0 || maximum == 0 || requested > maximum) return false;
    started = now; duration = requested; state = SPRAYING; return true;
  }
  void communication(uint32_t now) { lastCommunication = now; }
  void stop() { if (state != ESTOP && state != FAULT) state = DISARMED; }
  void emergencyStop() { state = ESTOP; }
  void fault() { state = FAULT; }
  void tick(uint32_t now, uint32_t timeout, bool emergency, bool faultDetected) {
    if (emergency) { emergencyStop(); return; }
    if (faultDetected) { fault(); return; }
    if (state == ESTOP || state == FAULT) return;
    if (uint32_t(now - lastCommunication) >= timeout) { stop(); return; }
    if (state == SPRAYING && uint32_t(now - started) >= duration) stop();
  }
  bool outputRequested() const { return state == SPRAYING; }
};
