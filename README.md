# Brain MRI Tumor Detection AI

This project is a deep learning-based application for detecting brain tumors from MRI images. It uses a Convolutional Neural Network (CNN) built on **ResNet18** and provides an easy-to-use **Gradio web interface** for inference.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Inference](#inference)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview
Brain tumors can be life-threatening, and early detection is crucial. This AI model classifies brain MRI images into two categories:
- **Tumor**
- **No Tumor**

The model was trained on a dataset of MRI images with preprocessing steps including resizing, normalization, and channel adjustment.  

---

## Features
- CNN model based on **ResNet18**
- Handles **single-channel grayscale MRI images**
- Training and test split with **PyTorch DataLoader**
- Real-time inference via **Gradio web app**
- Returns **prediction label** and **confidence score**
- Easy-to-use Python API for inference

---

## Dataset
- Total images: 253  
  - **No Tumor:** 98 images  
  - **Tumor:** 155 images  
- Preprocessing: resizing, normalization, and conversion to 3-channel input (if needed)

> Note: Dataset is expected to be structured as follows:
data/
├── BrainMRI/
│ ├── no/
│ │ ├── 1_no.jpeg
│ │ └── ...
│ └── yes/
│ ├── Y1.jpg
│ └── ...


---

## Installation

Clone the repository:

git clone https://github.com/yourusername/brain_mri_ai.git
cd brain_mri_ai

pip install -r requirements.txt

python -m src.app

Upload a brain MRI image

Click Submit

See the Prediction and Confidence Score



