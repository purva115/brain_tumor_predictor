import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Define image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((128, 128)),       # Resize all images to 128x128
    transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels for ResNet
    transforms.ToTensor(),               # Convert image to tensor
    transforms.Normalize(                # Normalize using ImageNet stats
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def process_image(image_path):
    """
    Preprocess a single MRI image
    Args:
        image_path (str): Path to the MRI image
    Returns:
        torch.Tensor: Preprocessed image tensor ready for model input
    """
    # Open image
    img = Image.open(image_path).convert("RGB")
    # Apply preprocessing
    img_tensor = preprocess(img)
    # Add batch dimension: [1, 3, 128, 128]
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor
