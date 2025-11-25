# Brain Tumor Predictor
Installation

<<<<<<< HEAD
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

git clone (https://github.com/purva115/brain_tumor_predictor.git)
cd brain_mri_ai
pip install -r requirements.txt
python -m src.inference.test_inference [to test from backend]
python -m src.app [to run with frontend]
Upload a brain MRI image
Click Submit
See the Prediction and Confidence Score
=======
Clone the repository and enter the project directory:
git clone https://github.com/purva115/brain_tumor_predictor.git
cd brain_tumor_predictor

Install required packages:
pip install -r requirements.txt

Quick backend test:
python -m src.inference.test_inference

Run the application (frontend + backend):
python -m src.app

How to use
1. Upload a brain MRI image in the web UI.
2. Click "Submit".
3. View the predicted class and the confidence score.

A compact deep-learning project to detect and classify brain tumors from MRI scans. This repository contains code for training, evaluating, and running inference with a convolutional neural network (or other models) on a brain MRI dataset.

## Features
- Trainable image classification model for brain tumor detection
- Modular scripts for training, evaluation and inference
- Support for custom datasets and preprocessing pipelines
- Configurable hyperparameters and checkpoints
>>>>>>> 381c5bc (readme)

## Requirements
- Python 3.8+
- Packages listed in `requirements.txt` (example: torch/keras, torchvision/tensorflow, numpy, scikit-learn, opencv-python, matplotlib, pandas)
- GPU recommended for training

## Installation
1. Clone the repository:
    git clone <repo-url>
2. Create and activate a virtual environment:
    python -m venv .venv
    source .venv/bin/activate   # Linux/macOS
    .venv\Scripts\activate      # Windows
3. Install dependencies:
    pip install -r requirements.txt

## Dataset
- Use a labeled MRI dataset (e.g., public brain MRI datasets such as BraTS or other curated collections).
- Directory expected format (example):
  dataset/
     train/
        class_1/
        class_2/
     val/
        class_1/
        class_2/
     test/
        images/

- Implement or adapt a preprocessing script to resize, normalize, and augment images.

## Project Structure (suggested)
src/
  train.py         # training loop and checkpointing
  evaluate.py      # evaluation and metrics
  predict.py       # single-image or batch inference
  data.py          # dataset loader and transforms
  model.py         # model definitions
  utils.py         # helpers (logging, checkpoints, metrics)
requirements.txt
README.md
checkpoints/
dataset/
results/
notebooks/

## Usage

Train:
  python src/train.py --config configs/train.yaml

Evaluate:
  python src/evaluate.py --checkpoint checkpoints/best.pth --data dataset/test

Predict (single image):
  python src/predict.py --checkpoint checkpoints/best.pth --image path/to/image.jpg

Common CLI options:
  --batch-size
  --epochs
  --lr / --learning-rate
  --device (cpu|cuda)
  --input-size
  --checkpoint

Adjust paths and hyperparameters via a config file (YAML/JSON) or command-line flags.

## Metrics & Outputs
- Typical metrics: accuracy, precision, recall, F1-score, AUC
- Save best checkpoints, training logs, and evaluation reports under `results/` or `checkpoints/`.
- Optionally add Grad-CAM or saliency visualizations for explainability.

## Tips
- Start training with a small subset to ensure pipeline works.
- Use data augmentation and class balancing for better generalization.
- Monitor GPU usage and enable mixed precision (if using PyTorch) to speed up training.

## Contributing
- Open an issue or PR for bugs, features, or dataset additions.
- Follow the code style and add tests for new functionality.

## License
Specify a license (e.g., MIT). Add a `LICENSE` file.

## Contact
For questions or help, open an issue in the repository.

