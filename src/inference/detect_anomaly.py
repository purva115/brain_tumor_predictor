import torch
from torchvision import transforms
from PIL import Image
from src.models.cnn_classifier import BrainMRIClassifier

MODEL_PATH = "models/brain_tumor_model.pth"

model = BrainMRIClassifier(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_condition(image_path):
    image = Image.open(image_path).convert('RGB')
    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_class = torch.argmax(probs).item()
    return pred_class, probs.tolist()

