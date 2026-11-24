#pragma once
#include <stdint.h>

// These are intentionally unset. Measure and validate them on the actual hardware.
namespace HardwareConfig {
constexpr unsigned long BAUD_RATE = 115200;
constexpr uint32_t COMMUNICATION_TIMEOUT_MS = 5000;
constexpr bool PHYSICAL_ACTUATION_ENABLED = false;
constexpr bool CALIBRATION_VALIDATED = false;
constexpr bool APPLICATION_LIMITS_VALIDATED = false;
constexpr int RELAY_PIN = -1;       // No guessed GPIO assignment.
constexpr bool RELAY_POLARITY_VALIDATED = false;
constexpr bool RELAY_ACTIVE_LOW = true; // Ignored until polarity has been verified.
constexpr int EMERGENCY_STOP_PIN = -1;
constexpr int FAULT_INPUT_PIN = -1;
constexpr float FLOW_ML_PER_SECOND = 0;
constexpr float TARGET_VOLUME_ML = 0; // Approved independent target; never severity-derived.
constexpr float AVAILABLE_VOLUME_ML = 0;
constexpr float TANK_CAPACITY_ML = 0;
constexpr uint32_t MAXIMUM_DURATION_MS = 0;
// Record measured nozzle specification, pipe diameter, pump voltage, pressure,
// target coverage, product application constraints, and calibration evidence
// before a future hardware-enabled build. They are not dosage formulas.
}
