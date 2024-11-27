#!/bin/bash

sudo apt install tesseract-ocr
export TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata/
echo "Install tesseract-ocr for rag_dataloader.py"

chmod +x scripts/*

# Configure core.hooksPath to use the hooks directory
git config core.hooksPath scripts
echo "Git hooks path configured to use the 'scripts/' directory."

