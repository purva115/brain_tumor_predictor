import torch
import torch.nn as nn
from torchvision import models
from src.utils.preprocessing import process_image

class BrainTumorClassifier:
    def __init__(self, model_path, device=None):
        # Use GPU if available
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load pre-trained ResNet18 and modify the final layer
        self.model = models.resnet18(weights=None)  # don't load ImageNet weights
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)  # 2 classes: Tumor / No Tumor

        # Load trained weights
        state_dict = torch.load(model_path, map_location=self.device)
        # If your saved model has a "model." prefix, remove it
        if list(state_dict.keys())[0].startswith("model."):
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Class labels
        self.labels = ["No Tumor", "Tumor"]

    def predict(self, image_path):
        print(f"Predicting on image: {image_path}")
        """
        Predict tumor presence on a single MRI image.
        Returns:
            label (str): "Tumor" or "No Tumor"
            confidence (float): Probability of the predicted class
        """
        # Preprocess the image
        img_tensor = process_image(image_path).to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)
            label = self.labels[pred_idx.item()]

        return label, confidence.item()
