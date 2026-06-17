# YOLO Traffic Violation Detector - All 9 Categories

This is a real YOLO-based Streamlit project for the hackathon theme:

**Automated Photo Identification and Classification for Traffic Violations Using Computer Vision**

## Important Truth

Generic YOLO models such as `yolov8n.pt` detect common COCO objects such as person, car, motorcycle, bus, truck, bicycle, and traffic light. They do **not** automatically know special traffic-violation classes such as no-helmet, seatbelt violation, stop-line violation, wrong-side driving, or number plate unless you train a custom YOLO model.

So this project has two layers:

1. **YOLO COCO detection**: Works immediately for person/vehicle detection.
2. **Custom all-9 violation YOLO model support**: Put trained weights at:

```text
models/traffic_violation_yolo.pt
```

When this file is present, the app uses YOLO to detect all 9 categories directly.

## Supported 9 Outputs

1. Helmet violation
2. Triple riding
3. Illegal parking
4. Wrong-side driving
5. Red-light violation
6. Stop-line violation
7. Seatbelt violation
8. Number plate detection
9. Multiple violations

## Install on Windows CPU

Open PowerShell inside this folder and run:

```powershell
install_windows_cpu.bat
```

Then run:

```powershell
python -m streamlit run app.py
```

## Manual Install

```powershell
python -m pip uninstall -y torch torchvision torchaudio ultralytics numpy opencv-python
python -m pip install --upgrade pip
python -m pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python check_torch.py
python -m streamlit run app.py
```

## Custom Model Classes

Train a YOLO model with these labels:

```text
helmet
no_helmet
triple_riding
illegal_parking
wrong_side_driving
red_light_violation
stop_line_violation
seatbelt
no_seatbelt
license_plate
multiple_violations
```

After training, copy your best model:

```text
runs/detect/traffic_violation_yolo_all9/weights/best.pt
```

to:

```text
models/traffic_violation_yolo.pt
```

## Train Command

Prepare a YOLO dataset and run:

```powershell
python train_custom_model.py
```

## For Presentation

Say this clearly:

> Our prototype uses YOLO for real object detection. The base YOLO model detects vehicles, people, motorcycles, and traffic lights. For true all-9 violation detection, the system is ready to load a custom-trained YOLO traffic violation model. This architecture is scalable because each violation class can be trained and improved using real traffic department image datasets.
