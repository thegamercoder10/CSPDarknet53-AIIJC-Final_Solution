# AIIJC - Final Solution (Road Sign Classification)

[![Result](https://img.shields.io/badge/Result-4th%20Place-brightgreen)](https://aiijc.com/en/results/)
[![Track](https://img.shields.io/badge/Track-AI%20in%20Geoservices-blue)]()
[![Model](https://img.shields.io/badge/Model-EfficientNetB2-orange)]()

This repository contains the final solution and deployment code for the **Team Round of the AIIJC Contest**. 
- **Team**: CSPDarknet53
- **Track**: AI in geoservices
- **Task**: Develop a Road Sign Classifier (Dataset provided by 2GIS). 

The project traces back to the [2gis/signs_classification_aij2021](https://github.com/2gis/signs_classification_aij2021) competition task.

## Overview

The core objective was to develop a multiclass-multilabel image classification model to categorize road signs across 6 basic headings. The model classifies images into 131 distinct classes. We utilized an **EfficientNetB2** architecture due to its optimal balance between accuracy and computational efficiency. 

To improve generalization and robustness, we implemented a comprehensive data augmentation pipeline using `albumentations` (including optical distortion, color jitter, random shadow, and blur).

## Repository Structure

The repository is divided into two main components:

### 1. Solution Source (Training & Pipeline)
Located in the `Solution Source (Source Code)/` directory.
- `pipeline/`: Contains the modularized PyTorch training pipeline.
  - `dataset.py` & `utils.py`: Data loading, preprocessing, and augmentation logic.
  - `models.py`: Model definitions and architecture setup.
  - `train.py`: The main training loop.
  - `generate_submission.py`: Script to generate predictions on the test set.
- `EfficientNetB2.ipynb` & `EfficientNetB2Final.ipynb`: Jupyter notebooks used for EDA, experimentation, and finalizing the model training process. 

### 2. Deployment (Web Interface & Bot)
Located in the `aiijc-web/` directory.
- **Flask Web Application** (`sitemain.py`): A lightweight web application exposing a user interface for image uploads and real-time predictions. Uses `flask-ngrok` for easy public access.
- **Inference Engine** (`inference.py`): Handles the loading of model weights (`finalw.pth`), preprocessing uploaded images, model inference, and translating predicted classes from Russian to English.
- **Telegram Bot** (`telegram-bot.ipynb`): An alternative chat-based interface for users to send images to a bot and receive classification results directly in Telegram.

## How to Run

### Web Application
1. Navigate to the `aiijc-web` directory.
2. Ensure you have the trained weights file (`finalw.pth`) and the class mapping file (`class2label.json`) in the directory.
3. Install dependencies: `pip install flask flask-ngrok torch torchvision albumentations opencv-python pillow efficientnet_pytorch`
4. Run the app: 
   ```bash
   python sitemain.py
   ```
5. An ngrok URL will be generated to access the web interface publicly.

### Telegram Bot
1. Navigate to the `aiijc-web` directory.
2. Open `telegram-bot.ipynb` in Jupyter Notebook.
3. Insert your Telegram Bot Token.
4. Run the notebook cells to start polling.

## Result
Our solution achieved **4th place** in the Finals. 

*For the original dataset and baseline, refer to the [2GIS AIJ2021 Repository](https://github.com/2gis/signs_classification_aij2021).*
