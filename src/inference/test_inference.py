from src.inference.inference_pipeline import BrainTumorClassifier

classifier = BrainTumorClassifier("models/brain_tumor_model.pth")
# label, confidence = classifier.predict("data/BrainMRI/no/33 no.jpg")
label, confidence = classifier.predict("data/BrainMRI/yes/Y18.jpg")
print(f"Prediction: {label}, Confidence: {confidence:.2f}")
