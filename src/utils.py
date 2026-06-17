from pathlib import Path
from datetime import datetime
import cv2
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = BASE_DIR / 'data' / 'uploads'
OUTPUT_DIR = BASE_DIR / 'outputs'
CSV_PATH = BASE_DIR / 'data' / 'violations.csv'

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH.parent.mkdir(parents=True, exist_ok=True)


def save_uploaded_file(uploaded_file):
    safe_name = uploaded_file.name.replace(' ', '_')
    path = UPLOAD_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name}"
    path.write_bytes(uploaded_file.getbuffer())
    return path


def read_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError('Could not read image. Please upload JPG/PNG image.')
    return image


def append_records(records):
    if not records:
        return
    df = pd.DataFrame(records)
    if CSV_PATH.exists():
        old = pd.read_csv(CSV_PATH)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)
