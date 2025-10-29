import gradio as gr
from src.inference.inference_pipeline import BrainTumorClassifier

# Load your trained model
model_path = "models/brain_tumor_model.pth"
classifier = BrainTumorClassifier(model_path)

def predict_tumor(image_path):
    print("Received input:", image_path)
    label, confidence = classifier.predict(image_path)  # Gradio gives a temporary file
    print(f"Prediction: {label}, Confidence: {confidence:.2f}")
    return f"Prediction: {label}", f"Confidence: {confidence:.2f}"

# Build Gradio interface
interface = gr.Interface(
    fn=predict_tumor,
    inputs=gr.Image(type="filepath", label="Upload Brain MRI"),
    outputs=[gr.Textbox(label="Prediction"), gr.Textbox(label="Confidence")],
    title="AI Brain Tumor Detection",
    description="Upload a brain MRI image to detect if a tumor is present.",
    allow_flagging="never"  # optional
)

# Launch the app
interface.launch()
