import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from config import DEVICE, INPUT_SIZE, IDX2EMOTION, TRAINED_MODEL_PATH
from models.emotion_model import load_model, GradCAM

# Load a face detector (OpenCV Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


def crop_face(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        # fallback: full image (not ideal)
        return image_bgr

    # Use the largest detected face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Add a small margin to include chin/forehead
    margin = int(0.15 * h)
    y0 = max(0, y - margin)
    y1 = min(image_bgr.shape[0], y + h + margin)
    x0 = max(0, x - margin)
    x1 = min(image_bgr.shape[1], x + w + margin)

    face_crop = image_bgr[y0:y1, x0:x1]
    return face_crop


def predict_image(image_path: str, output_cam_path: str = None):
    model = load_model(TRAINED_MODEL_PATH, DEVICE)
    gradcam = GradCAM(model)

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image {image_path}")

    # 🔥 KEY FIX: crop face first
    face_bgr = crop_face(img_bgr)

    img_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    input_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_label = IDX2EMOTION[pred_idx]

    # GradCAM
    cam = gradcam.generate(input_tensor)
    cam_resized = cv2.resize(cam, (face_bgr.shape[1], face_bgr.shape[0]))
    heatmap = (cam_resized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = (0.4 * heatmap + 0.6 * face_bgr).astype(np.uint8)

    gradcam_path = None
    if output_cam_path:
        cv2.imwrite(output_cam_path, overlay)
        gradcam_path = output_cam_path

    return pred_label, probs, gradcam_path
