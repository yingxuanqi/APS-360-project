# APS360 Project – Agricultural Anomaly Detection from Aerial Images

## Overview
This project focuses on detecting abnormal agricultural regions from aerial imagery using convolutional neural networks (CNNs). The task is formulated as a binary classification problem, where each image is classified as either **normal** or **abnormal**.

The abnormal class includes agricultural regions with visible issues such as water-related damage or weed clusters. The project compares a baseline RGB model with a primary model that uses both **RGB and NIR** information.

---

## Task
- **Input:** Aerial images from the Agriculture-Vision dataset  
- **Output:** Binary label  
  - `0` = normal  
  - `1` = abnormal  

The goal is to determine whether adding NIR information improves abnormal-region detection compared to using RGB alone.

---

## Dataset
The dataset is based on aerial farmland images and corresponding labels stored in CSV files.

### Data used in the primary model
- RGB images
- NIR images
- Binary labels from `labels.csv`

### Preprocessing
- RGB images are loaded as 3-channel images
- NIR images are loaded as 1-channel grayscale images
- RGB and NIR are concatenated into a **4-channel tensor**
- All images are resized to **224 × 224**

---

## Primary Model
The primary model is a 4-channel convolutional neural network that takes RGB + NIR as input.

### Architecture
- Input: `4 × 224 × 224`
- Conv2d: `4 → 16`, kernel size `3`, padding `1`
- ReLU
- MaxPool2d: `2 × 2`
- Conv2d: `16 → 32`, kernel size `3`, padding `1`
- ReLU
- MaxPool2d: `2 × 2`
- Border cropping on feature maps
- Fully connected layer: `32 × 48 × 48 → 128`
- ReLU
- Output layer: `128 → 2`

### Design note
After the second pooling layer, the outer border of the feature map is removed before flattening. This is intended to reduce the influence of edge regions and make the model focus more on the central agricultural area.

---

## Training Setup
- Loss function: `CrossEntropyLoss`
- Optimizer: `Adam`
- Learning rate: `1e-4`
- Batch size: `32`
- Number of epochs: `10`

The best model is saved based on validation accuracy.

---

## Grad-CAM Visualization
Grad-CAM is used to visualize which spatial regions contribute most to the model’s prediction.

For each selected validation sample, the script displays:
1. RGB image
2. NIR image
3. Grad-CAM heatmap
4. Heatmap overlay on the RGB image

This helps qualitatively evaluate whether the model attends to meaningful abnormal regions.

---

## Repository Structure
- `baseline.py`  
  Baseline RGB-only CNN

- `primary(training + heatmap).py`  
  Primary 4-channel CNN with training and Grad-CAM visualization

- `dataprocess.py`  
  Data processing and preparation scripts

- `label maker.py`  
  Label generation script

- `testing.py`  
  Model evaluation / testing utilities

- `Baseline heatmap/`  
  Saved baseline heatmap outputs

---

## Expected Contribution
The main contribution of this project is to evaluate whether adding NIR information improves abnormal agricultural region detection compared to an RGB-only baseline.

The project also includes qualitative interpretation through Grad-CAM to analyze model attention.

---

## Limitations
- The task is currently simplified to binary classification
- Performance may depend on image quality and dataset composition
- The dataset may not fully represent all agricultural environments
- Cropping edge regions may remove some useful information in certain cases

---

## Future Work
Possible future improvements include:
- Testing on fully unseen external data
- Extending the task to multi-label classification
- Using a deeper CNN or transfer learning model
- Improving robustness under low-quality or complex-background images

---

## Author
Yingxuan Qi
