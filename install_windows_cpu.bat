@echo off
echo ================================================
echo Installing compatible CPU Torch + YOLO packages
echo ================================================
python -m pip uninstall -y torch torchvision torchaudio ultralytics numpy opencv-python
python -m pip install --upgrade pip
python -m pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
echo.
echo Checking TorchVision NMS...
python check_torch.py
echo.
echo If check is OK, run: python -m streamlit run app.py
pause
