from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parents[1]

VIOLATION_LABELS = {
    'helmet': 'Helmet worn',
    'no_helmet': 'Helmet Violation',
    'helmet_violation': 'Helmet Violation',
    'triple_riding': 'Triple Riding',
    'illegal_parking': 'Illegal Parking',
    'wrong_side_driving': 'Wrong-side Driving',
    'red_light_violation': 'Red-light Violation',
    'stop_line_violation': 'Stop-line Violation',
    'seatbelt': 'Seatbelt worn',
    'no_seatbelt': 'Seatbelt Violation',
    'seatbelt_violation': 'Seatbelt Violation',
    'license_plate': 'Number Plate Detected',
    'number_plate': 'Number Plate Detected',
    'multiple_violations': 'Multiple Violations',
}

COCO_VEHICLE_CLASSES = {'car', 'motorcycle', 'bus', 'truck', 'bicycle'}
COCO_PERSON_CLASS = 'person'
COCO_TRAFFIC_LIGHT_CLASS = 'traffic light'

@dataclass
class Detection:
    label: str
    conf: float
    box: Tuple[int, int, int, int]
    source: str

class TrafficViolationDetector:
    def __init__(self, config_path: str = 'config.yaml'):
        self.config = yaml.safe_load((BASE_DIR / config_path).read_text())
        self.conf = float(self.config.get('confidence', 0.25))
        self.iou = float(self.config.get('iou', 0.45))
        self.output_dir = BASE_DIR / self.config.get('output_dir', 'outputs')
        self.output_dir.mkdir(exist_ok=True)

        # Generic YOLO model: detects COCO objects like person, motorcycle, car, bus, truck, traffic light.
        # If yolov8n.pt is not present, Ultralytics downloads it automatically on first run.
        self.vehicle_model = YOLO(self.config.get('vehicle_model', 'yolov8n.pt'))

        # Custom model: this is the correct way to detect all 9 violation classes directly by YOLO.
        # Put your trained weights at models/traffic_violation_yolo.pt.
        custom_path = BASE_DIR / self.config.get('custom_violation_model', 'models/traffic_violation_yolo.pt')
        self.custom_model = YOLO(str(custom_path)) if custom_path.exists() else None

    def _predict(self, model: YOLO, image: np.ndarray, source: str) -> List[Detection]:
        results = model.predict(image, conf=self.conf, iou=self.iou, verbose=False)
        detections: List[Detection] = []
        for r in results:
            names = r.names
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = str(names.get(cls_id, cls_id)).lower().replace(' ', '_')
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append(Detection(label, conf, (x1, y1, x2, y2), source))
        return detections

    @staticmethod
    def _overlap_ratio(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        return inter / area_a

    @staticmethod
    def _expand(box, image_shape, scale=0.35):
        h, w = image_shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        return (
            max(0, int(x1 - bw * scale)),
            max(0, int(y1 - bh * scale)),
            min(w - 1, int(x2 + bw * scale)),
            min(h - 1, int(y2 + bh * scale)),
        )

    def _rule_based_candidates_from_coco(self, detections: List[Detection], image_shape) -> List[Detection]:
        """Uses YOLO COCO detections + rules to produce candidates.
        This is not as accurate as a custom trained violation model, but it keeps the app YOLO-based.
        """
        h, w = image_shape[:2]
        people = [d for d in detections if d.label == COCO_PERSON_CLASS]
        motorcycles = [d for d in detections if d.label == 'motorcycle']
        vehicles = [d for d in detections if d.label in COCO_VEHICLE_CLASSES]
        traffic_lights = [d for d in detections if d.label == COCO_TRAFFIC_LIGHT_CLASS.replace(' ', '_') or d.label == 'traffic_light']

        candidates: List[Detection] = []

        # Triple riding and helmet candidate on motorcycles.
        for m in motorcycles:
            expanded = self._expand(m.box, image_shape, 0.7)
            near_people = [p for p in people if self._overlap_ratio(p.box, expanded) > 0.05]
            if len(near_people) >= 3:
                candidates.append(Detection('triple_riding', 0.76, m.box, 'YOLO+rule'))
            if len(near_people) >= 1:
                # Generic COCO YOLO cannot see helmet class, so mark as candidate.
                candidates.append(Detection('no_helmet_candidate', 0.55, near_people[0].box, 'YOLO+rule'))

        # Illegal parking candidate: vehicle near side/bottom road region.
        for v in vehicles:
            x1, y1, x2, y2 = v.box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            if center_y > 0.58 * h and (center_x < 0.25 * w or center_x > 0.75 * w):
                candidates.append(Detection('illegal_parking', 0.58, v.box, 'YOLO+rule'))

        # Red-light / stop-line candidates need traffic signal and lane-line context.
        if traffic_lights and vehicles:
            front_vehicle = max(vehicles, key=lambda d: d.box[3])
            candidates.append(Detection('red_light_violation', 0.52, front_vehicle.box, 'YOLO+rule'))
            candidates.append(Detection('stop_line_violation', 0.50, front_vehicle.box, 'YOLO+rule'))

        # Wrong-side and seatbelt are difficult from generic COCO single image.
        # If a custom model is absent, show only when a large vehicle/person context exists as candidate.
        if len(vehicles) >= 1 and len(people) >= 1:
            v = vehicles[0]
            candidates.append(Detection('seatbelt_violation_candidate', 0.45, v.box, 'YOLO+rule'))
        if len(vehicles) >= 2:
            candidates.append(Detection('wrong_side_driving_candidate', 0.45, vehicles[0].box, 'YOLO+rule'))

        return candidates

    def _draw(self, image: np.ndarray, detections: List[Detection], violations: List[Dict]) -> np.ndarray:
        out = image.copy()

        for d in detections:
            x1, y1, x2, y2 = d.box
            label = d.label.replace('_', ' ')
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 180, 255), 2)
            cv2.putText(out, f'{label} {d.conf:.2f}', (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 2)

        for v in violations:
            x1, y1, x2, y2 = v['box']
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(out, f"{v['violation_type']} {v['confidence']}", (x1, max(25, y1 - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        return out

    def detect(self, image: np.ndarray, filename: str) -> Dict:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        coco_dets = self._predict(self.vehicle_model, image, 'YOLO-COCO')
        custom_dets: List[Detection] = []
        if self.custom_model is not None:
            custom_dets = self._predict(self.custom_model, image, 'YOLO-custom')

        violations: List[Dict] = []

        # 1) Direct custom YOLO violation detections.
        for d in custom_dets:
            key = d.label.lower().replace(' ', '_')
            if key in VIOLATION_LABELS and 'worn' not in VIOLATION_LABELS[key].lower():
                violations.append({
                    'timestamp': timestamp,
                    'image': filename,
                    'violation_type': VIOLATION_LABELS[key],
                    'confidence': round(d.conf, 2),
                    'source': d.source,
                    'box': d.box,
                })

        # 2) YOLO COCO + rule candidates when custom all-9 model is not present.
        if self.custom_model is None:
            for d in self._rule_based_candidates_from_coco(coco_dets, image.shape):
                label_map = {
                    'no_helmet_candidate': 'Helmet Violation Candidate',
                    'seatbelt_violation_candidate': 'Seatbelt Violation Candidate',
                    'wrong_side_driving_candidate': 'Wrong-side Driving Candidate',
                    'triple_riding': 'Triple Riding',
                    'illegal_parking': 'Illegal Parking Candidate',
                    'red_light_violation': 'Red-light Violation Candidate',
                    'stop_line_violation': 'Stop-line Violation Candidate',
                }
                violations.append({
                    'timestamp': timestamp,
                    'image': filename,
                    'violation_type': label_map.get(d.label, d.label),
                    'confidence': round(d.conf, 2),
                    'source': d.source,
                    'box': d.box,
                })

        annotated = self._draw(image, coco_dets + custom_dets, violations)
        out_path = self.output_dir / f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{Path(filename).stem}.jpg"
        cv2.imwrite(str(out_path), annotated)

        vehicles = [d for d in coco_dets if d.label in COCO_VEHICLE_CLASSES]
        people = [d for d in coco_dets if d.label == 'person']

        status = {
            'generic_yolo_coco_loaded': True,
            'custom_all_9_violation_yolo_loaded': self.custom_model is not None,
            'custom_model_expected_path': 'models/traffic_violation_yolo.pt',
            'note': 'For true all-9 automatic violation detection, train/add models/traffic_violation_yolo.pt. Without it, the app uses YOLO COCO + rule-based candidates.'
        }

        return {
            'annotated_image': annotated,
            'output_path': str(out_path),
            'total_objects': len(coco_dets) + len(custom_dets),
            'vehicles_detected': len(vehicles),
            'people_detected': len(people),
            'violations_detected': len(violations),
            'violations': violations,
            'model_status': status,
            'all_detections': [d.__dict__ for d in coco_dets + custom_dets],
        }
