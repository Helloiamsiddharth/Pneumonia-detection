# Pneumonia Detection using Deep Learning

A deep learning-based system to detect **Pneumonia** from chest X-ray images using Convolutional Neural Networks (CNNs).  
The model classifies chest X-rays into two classes: **NORMAL** and **PNEUMONIA**.

## Project Overview

Pneumonia is a serious lung infection that can be life-threatening, especially in children and the elderly. Early and accurate diagnosis is critical.  
This project uses **chest X-ray images** and modern deep learning techniques to assist radiologists in detecting pneumonia automatically.

### Key Features
- Binary classification (Normal vs Pneumonia)
- Transfer learning with pre-trained CNN backbones
- Data augmentation to handle class imbalance and small dataset size
- Model evaluation with **accuracy**, **precision**, **recall**, **F1-score**, **ROC-AUC**
- Grad-CAM visualizations for model interpretability
- web application using streamlit

## Dataset

The model is trained on the widely-used **Chest X-Ray Images (Pneumonia)** dataset from Kaggle.

- **Source**: [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Classes**: NORMAL, PNEUMONIA
- **Training images**: ~5,216
- **Validation images**: ~16
- **Test images**: ~624

> **Note**: The dataset is quite imbalanced (more pneumonia cases than normal). We use data augmentation and class weighting to mitigate this.

## Model Architecture

### Best Performing Model (as of now)
- Backbone: **EfficientNetB3** / **ResNet50** / **DenseNet121**
- Custom head: Global Average Pooling → Dense layers → Sigmoid output
- Input size: 224×224 or 300×300 
- Optimizer: **Adam**
- Loss: **Binary Crossentropy**
- Regularization: Dropout, Batch Normalization, data augmentation

Alternative models tested: VGG16, MobileNetV2, Xception, custom CNN from scratch.
