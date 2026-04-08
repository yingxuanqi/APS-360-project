# APS-360-project
# APS360 Project – Agricultural Anomaly Detection

## Overview
This project aims to detect abnormal agricultural regions (e.g., waterlogging and weed clusters) from aerial images using convolutional neural networks (CNNs).

The goal is to improve detection performance compared to a baseline model and analyze model behavior using visualization techniques.

---

## Dataset
The dataset is derived from the Agriculture-Vision dataset.

- Input: Aerial images (RGB or RGB + NIR)
- Output: Binary classification (Normal vs Abnormal)

Images are resized to 224×224 before being fed into the model.

---

## Models

### Baseline Model
- Input: RGB images (3 channels)
- Architecture: Simple CNN
- Purpose: Provide a reference for comparison

### Primary Model
- Input: RGB + NIR (4 channels)
- Architecture: CNN (same structure as baseline)
- Key Idea: Use NIR information to improve detection of subtle agricultural patterns

---

## Visualization (Heatmaps)
Grad-CAM heatmaps are generated to visualize where the model focuses when making predictions.

- Baseline model: Often shows diffuse attention
- Primary model: Produces more localized and meaningful attention

---

## File Structure

- `baseline.py`  
  Training and evaluation of the baseline CNN (3-channel input)

- `primary(training + heatmap).py`  
  Training and heatmap generation for the primary model (4-channel input)

- `primary1.py`  
  Alternative or testing version of the primary model

- `dataprocess.py`  
  Data loading and preprocessing

- `label maker.py`  
  Script for generating labels

- `testing.py`  
  Model evaluation on test data

- `Baseline heatmap/`  
  Visualization outputs for the baseline model

---

## Results

The primary model improves performance compared to the baseline:

- Higher recall and F1 score
- Improved detection of abnormal regions
- More focused attention in Grad-CAM visualization

---

## Limitations

- Performance drops on low-quality images
- Dataset bias due to removal of roads/buildings
- Limited generalization to unseen environments

---

## Conclusion

Adding NIR information improves the model’s ability to detect subtle agricultural anomalies, but further work is needed to improve robustness and generalization.

---
