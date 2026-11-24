# Rice Leaf AI: classification, segmentation, reviewed severity and ESP32 simulation

The application reuses the existing trained ResNet-18 and U-Net checkpoints. It never trains models or writes into existing evaluation folders.

## Launch

From this project folder:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, normally http://localhost:8501. Alternatively double-click `run_app.bat`. The current environment already has the core app dependencies. Installation of `pyserial` and `pillow-heif` was blocked by network restrictions during implementation; install them on a network-enabled computer for USB and HEIC/HEIF support:

```powershell
python -m pip install pyserial pillow-heif
```

JPEG/PNG, trained inference, the dashboard, and mock-device simulation work without these two optional capabilities. For iPhone input, export JPEG when HEIF decoding is unavailable. EXIF orientation is corrected. Uploads are limited to 30 MB and 25 megapixels; export a smaller photo if needed. The host computer runs Python; the iPhone supplies photos.

PyTorch and torchvision must be a compatible pair for your Python/OS. The implementation was tested with Python 3.13.6, torch 2.15.0+cpu, torchvision 0.29.2+cpu, Streamlit 1.61.1, numpy 2.3.2, pandas 3.0.1 and Pillow 11.3.0. `requirements.txt` specifies allowed ranges, not a universally interchangeable torch/torchvision combination. Existing CUDA-trained state dictionaries load on CPU.

## Rebuild the training data

### Download and extract the original dataset

1. Download the original [rice-leaf dataset from Google Drive](https://drive.google.com/drive/folders/1GMvTcJs96Uq2QNJ0efY7U2PCWCffIgLd?usp=sharing).
2. Extract the archive so the class folders are directly under `Dataset/Images/`
   in the project root. The expected layout is
   `Dataset/Images/<class>/<image>`. Keep the original class names, filenames,
   and image contents so they match the saved split manifest.

### Generate all training data

Run these commands from the project root:

```powershell
python -m pip install -r requirements.txt
python -m pip install matplotlib
python scripts/prepare_data.py --unet
```

The last command verifies the source images against the saved split and
regenerates all four outputs:

| Output folder | Contents |
| --- | --- |
| `cleaned dataset/` | Deduplicated originals. |
| `split_dataset/` | The original train, validation, and test splits. |
| `augmented_train/` | Augmented training images. |
| `results/pixel_annotations_full/` | Original and augmented U-Net draft masks and manifests. |

For just the cleaned and split images, run
`python scripts/prepare_data.py --images-only`. To also build the augmented
images, run `python scripts/prepare_data.py`.

The original dataset is ignored by Git. Keep the generated datasets local as
well and stage the app files explicitly when preparing a GitHub commit.

The U-Net notebook can use the generated masks
for a new experiment. Set the notebook's `OUTPUT` to a fresh directory and
set `RUN_TRAINING = True` to keep its results separate from the web app.

Commit the two active checkpoints with the app so a clone can run
inference after installing dependencies.

## Use the three tabs

1. **Disease Classification:** upload a rice leaf photo and select **Classify image**. Classification uses the frozen-backbone ResNet-18 checkpoint. Class names preserve the exact saved mapping, including `Healthy Rice Leaf`, `Leaf scald` and `Rice Turgro`. Confidence is a model probability, not severity or guaranteed correctness.
2. **Segmentation & Severity:** select **Segment diseased regions**. The app automatically extracts a whole-leaf mask from the uploaded photo, then runs U-Net with its original RGB preprocessing to produce the disease mask. No class label, leaf mask or Grad-CAM mask is fed into U-Net.
3. The app shows the automatic whole-leaf mask followed immediately by the calculated affected-area percentage and pixel counts. There is no mask uploader, blue overlay, or confirmation checkbox. The formula is `100 * count(disease AND leaf) / count(leaf)`. The displayed number is labeled an **automatic estimate**, because leaf extraction has not been independently validated. Empty leaf masks still show unavailable area. Display estimates do not become validated spray-control inputs.
4. **Model Evaluation:** view saved classification metrics, histories, matrices, U-Net scores/curves, and frozen-model Grad-CAM analysis. No full-dataset evaluation runs when this page opens.

All existing segmentation labels are automatic drafts pending review. Report U-Net scores as **draft-label agreement**. Reviewed leaf masks improve denominator provenance; they do not establish expert accuracy of the predicted disease numerator. Multiple leaves use the union of visible tissue; hidden/occluded area cannot be measured. Empty or unreviewed denominators produce unavailable severity, never an image-area substitute.

Healthy-class handling is conservative: raw U-Net output remains visible, a disagreement requires review, and Healthy blocks every spray simulation arm/cycle. It does not force an artificial empty disease mask or certify disease absence.

## Serial simulation and USB connection

The Python host runs Frozen ResNet-18 classification and U-Net segmentation.
In **Segmentation & Severity**, the **ESP32 connection and simulation** panel
uses `hardware/serial_communication.py` to send a prediction report and display
the device acknowledgement.

| Connection mode | Where the report goes |
| --- | --- |
| **Simulator** | A software ESP32 emulator in `hardware/simulator.py` on the same computer. It reads the mock settings from `config/hardware.json`. |
| **USB ESP32** | The board running `hardware/esp32_firmware/esp32_firmware.ino`, connected by a USB data cable. The Python host uses pyserial and the serial settings in `config/app.json`. |

For USB, flash the firmware described below, connect the board, close other
serial monitors, and enter its detected COM port. The default baud rate is
**115200**. Select **Connect**, then **Send prediction report** after processing
an image. The app displays the board's acknowledgement and can request its
status. Each browser session reuses its connection across reruns. Both modes
support simulation commands. Physical pump actuation is disabled.

- **Connect** performs a handshake, never an arm or spray.
- **Send prediction report** sends only class, confidence, valid severity (or null), and identifiers.
- **Arm simulation** and **Run one simulated cycle** are separate deliberate actions. A new valid report is needed after a simulated cycle. Healthy, invalid/no-leaf, stale, or zero-severity predictions are rejected.
- **Read device status** refreshes telemetry. There is no background heartbeat. After 5 seconds without a valid command the device disarms and clears its prediction; resend a report before another arm attempt.
- **Emergency stop** latches the stop. Reset the device to clear it; closing/reopening the mock transport creates a new simulated device.
- Connection loss, malformed replies, timeouts, resets and unmatched acknowledgements close the host connection. There are no automatic command retries or automatic reconnections.

`config/hardware.json` intentionally has null calibration/application values. Connection and prediction acknowledgement work immediately, but **spray planning/arming is blocked until those inputs exist**. The mock simulator accepts explicitly configured measured flow and an independently authorized target volume. It computes `duration_ms = target_volume_ml / flow_ml_per_second * 1000`, enforces volume/duration bounds, and returns simulation telemetry. **There is no severity-to-dose formula.**

## ESP32 firmware

Files are in `hardware/esp32_firmware/`. Open `esp32_firmware.ino` in the Arduino IDE, select the correct ESP32 board/port, and install ArduinoJson 7 plus the ESP32 board package. A PlatformIO configuration for a generic ESP32 Dev Module is included; verify your actual board before building. The ArduinoJson usage follows the official [deserialization](https://arduinojson.org/v7/api/json/deserializejson/) and [serialization](https://arduinojson.org/v7/api/json/serializejson/) APIs.

For Stage A, keep the pump electrically disconnected. The supplied firmware has **no relay-ON command or relay-ON path**. `HardwareConfig.h` leaves GPIO assignments unset, physical operation disabled, and calibration unapproved. USB startup logs are discarded before the protocol handshake. Do not mix debug text into the protocol serial channel.

The firmware returns null spray parameters until independently measured settings and authorized application limits are configured. Host JSON settings configure the mock device; they do not program the ESP32. Firmware settings must be recorded in `HardwareConfig.h` for a board build. Do not guess active-low/active-high relay behavior or pins. Hardware wiring/pull resistors must enforce OFF even before firmware startup and during reset.

`SafetyController.h` is a separate Stage B scaffold with explicit arm, bounded start, timeout, emergency stop, fault latching, rollover-safe timing and default OFF. It is tested natively but is **not connected to a physical actuation serial command**. A future hardware-enabled integration still needs verified interlocks, calibrated flow/coverage, actual board tests, target positioning, approved application limits, and a reviewed output backend. Changing an application JSON flag cannot enable a pump.

The ESP32 sketch was not compiled/flashed here because Arduino/PlatformIO and the ESP32 toolchain are unavailable. Native C++ safety-core tests are not a substitute for an ESP32 build or electrical tests. No physical hardware tests or pump actuation were performed.

## Protocol

Protocol version 1 uses bounded newline-delimited JSON, random session/message identifiers, increasing sequence numbers, device boot IDs, matched acknowledgements and replay rejection. Only simulation commands are exposed by the Python client schema. Opening/refreshing a page never sends a simulation or spray command.

## Treatment lookup sources

The classification page displays treatment examples from agricultural guidance. The lookup supplies no dosage and does not control the ESP32.

| Exact model label | Displayed treatment | Source |
| --- | --- | --- |
| Bacterial Leaf Blight | Copper hydroxide | [TNAU April 2026 advisory](https://agritech.tnau.ac.in/crop_protection/pdf/2026/04.%20Pest%20forecost%20April%202026.pdf) |
| Brown Spot | Mancozeb | [TNAU August 2024 advisory](https://agritech.tnau.ac.in/crop_protection/pdf/2024/05.%20Pest%20Forecast%20Report.pdf) |
| Healthy Rice Leaf | None required | No treatment for a healthy prediction. |
| Leaf Blast | Tricyclazole | [TNAU June 2024 advisory](https://agritech.tnau.ac.in/ta/crop_protection/pdf/2024/CPPS-Pest%20Forecast%20Report%20-June%202024.pdf) |
| Leaf scald | Validamycin | [TNAU leaf scald guidance](https://www.agritech.tnau.ac.in/expert_system/paddy/cpdisleafscald.html) |
| Narrow Brown Leaf Spot | Propiconazole | [IRRI and partners rice fact sheet](https://rkb-odisha.in/wp-content/uploads/2024/09/narrow-brown-spot-english.pdf) |
| Rice Hispa | Malathion | [TNAU June 2024 advisory](https://agritech.tnau.ac.in/ta/crop_protection/pdf/2024/CPPS-Pest%20Forecast%20Report%20-June%202024.pdf) |
| Rice Turgro | No pesticide cures tungro; manage leafhopper vectors | [IRRI Rice Tungro Disease Management](https://books.irri.org/9712200310_content.pdf), [PhilRice management guidance](https://dbmp.philrice.gov.ph/FAQs/src/search.php) |
| Sheath Blight | Validamycin | [TNAU rice crop protection guide](https://agritech.tnau.ac.in/crop_protection/crop_pdf/rice.pdf) |

## Files required to run the web app

| Files or folders | Purpose |
| --- | --- |
| `app.py`, `requirements.txt` | Streamlit entry point and Python dependencies. |
| `models/`, `inference/` | Network definitions, image preprocessing, and predictions. |
| `utils/`, `config/`, `hardware/` | Image upload helpers, app settings, and simulator and serial support. |
| `results/resnet18_frozen/best_model.pt` | Trained Frozen ResNet-18 classification checkpoint. |
| `results/unet_disease_segmentation/best_model.pt` | Trained U-Net segmentation checkpoint. |

For the simulator and USB ESP32 setup, see [Serial simulation and USB connection](#serial-simulation-and-usb-connection).

The **Model Evaluation** tab also uses the saved metrics, plots, and manifests in
`results/`, including `results/provenance/`. The original dataset, training
notebooks, and data-preparation scripts are needed for training or reproducing
experiments.

Run from the project root:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
