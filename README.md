# Personal Protective Equipment (PPE) Detection System

![PPE Detection Demo](demo.gif) *Example detection output*

This project implements a computer vision system for detecting Personal Protective Equipment (PPE) items using deep learning. The solution includes both fundamental image processing techniques and an advanced deep learning pipeline for PPE classification.

## Key Features

- **Image Processing Fundamentals**: Blending, histogram equalization, brightness adjustment, flipping, rotation
- **Deep Learning Models**: 
  - EfficientNetV2B3 with fine-tuning
  - ResNet101 with fine-tuning
- **Multi-label Classification**: Detects 11 PPE-related classes simultaneously
- **Web Interface**: Gradio-based app for real-time inference
- **Performance Metrics**: Precision, recall, F1-score, and confusion matrix analysis

## Class Detection Capabilities

The system detects these 11 PPE-related classes:

| Class Name      | Description                     |
|-----------------|---------------------------------|
| Helmet          | Safety helmet detection         |
| Gloves          | Protective gloves detection     |
| Vest            | Safety vest detection           |
| Boots           | Safety boots detection          |
| Goggles         | Safety goggles detection        |
| Person          | Person detection                |
| no_helmet       | Missing helmet detection        |
| no_goggle       | Missing goggles detection       |
| no_gloves       | Missing gloves detection        |
| no_boots        | Missing boots detection         |
| none            | No PPE items detected           |

## Technical Approach



### 1. Deep Learning Architecture
- **Backbone**: EfficientNetV2B3 and ResNet101 pre-trained on ImageNet
- **Custom Head**:
  - Global Average Pooling
  - Batch Normalization
  - Dropout layers (0.2-0.3)
  - Dense layers (128-256 units)
- **Output**: 11-unit sigmoid layer for multi-label classification

### 2. Training Strategy
- **Focal Loss** (γ=2, α=0.25) to handle class imbalance
- **Data Augmentation**:
  - Random horizontal flipping
  - Brightness/contrast adjustments
  - Hue/saturation variation
- **Fine-tuning**: Unfreezed top layers after initial training
- **Callbacks**:
  - Early stopping (patience=5)
  - Learning rate reduction on plateau

## Results Summary

### Performance Metrics (After Fine-Tuning)

| Model           | Avg. Precision | Avg. Recall | Avg. F1-Score |
|-----------------|----------------|-------------|---------------|
| EfficientNetV2B3| 0.78           | 0.75        | 0.76          |
| ResNet101       | 0.81           | 0.77        | 0.79          |



## Installation

```bash
# Clone repository
git clone https://github.com/SAJJADGORAYA1/Personal_Protective_Equipment_Detection_Research.git


# Install dependencies
pip install -r requirements.txt