from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from src.detector import TrafficViolationDetector
from src.utils import CSV_PATH, append_records, read_image, save_uploaded_file

st.set_page_config(page_title='YOLO Traffic Violation Detector', page_icon='🚦', layout='wide')

st.title('🚦 YOLO Traffic Violation Detector - All 9 Categories')
st.caption('Real YOLO pipeline with COCO detection + optional custom all-9 violation YOLO model')

with st.sidebar:
    st.header('Detected Violation Categories')
    st.markdown('''
    The code supports these 9 outputs:
    1. Helmet violation
    2. Triple riding
    3. Illegal parking
    4. Wrong-side driving
    5. Red-light violation
    6. Stop-line violation
    7. Seatbelt violation
    8. Number plate detection
    9. Multiple violations
    ''')
    st.warning('For true automatic all-9 detection, add trained weights at `models/traffic_violation_yolo.pt`. Without custom weights, YOLO COCO + rules produce candidates.')

@st.cache_resource
def get_detector():
    return TrafficViolationDetector()

uploaded_file = st.file_uploader('Upload traffic image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image_path = save_uploaded_file(uploaded_file)
    image = read_image(image_path)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Original Image')
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

    with st.spinner('Running YOLO inference...'):
        detector = get_detector()
        result = detector.detect(image, uploaded_file.name)
        append_records(result['violations'])

    with col2:
        st.subheader('YOLO Annotated Evidence')
        st.image(cv2.cvtColor(result['annotated_image'], cv2.COLOR_BGR2RGB), use_column_width=True)

    st.subheader('Summary')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Total YOLO Objects', result['total_objects'])
    c2.metric('Vehicles', result['vehicles_detected'])
    c3.metric('People', result['people_detected'])
    c4.metric('Violation Outputs', result['violations_detected'])

    st.subheader('YOLO Model Status')
    st.json(result['model_status'])

    if result['violations']:
        st.subheader('Violation Report')
        df = pd.DataFrame(result['violations'])
        st.dataframe(df.drop(columns=['box'], errors='ignore'), use_container_width=True)
    else:
        st.success('YOLO did not find any violation candidates in this image.')

    with st.expander('View all raw YOLO detections'):
        st.dataframe(pd.DataFrame(result['all_detections']), use_container_width=True)

    st.download_button(
        'Download Annotated Evidence Image',
        data=Path(result['output_path']).read_bytes(),
        file_name=Path(result['output_path']).name,
        mime='image/jpeg'
    )

st.divider()
st.subheader('Stored Violation Records')
if CSV_PATH.exists():
    try:
        records = pd.read_csv(CSV_PATH)
        st.dataframe(records.tail(50).drop(columns=['box'], errors='ignore'), use_container_width=True)
    except Exception:
        st.write('No records yet.')
else:
    st.write('No records yet.')
