#!/usr/bin/env bash
# exit on error
set -o errexit

pip install --upgrade pip
pip install -r requirements.txt

# If you use Tesseract OCR, you'd add apt-get commands here, 
# but Render's native environment usually handles basic python libs.