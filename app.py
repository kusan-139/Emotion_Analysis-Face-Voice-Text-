import os
import io
import sys

# --- FFMPEG PORTABLE SETUP (Start) ---
# This tells Python to look in your 'ffmpeg_bin' folder for commands
project_dir = os.path.dirname(os.path.abspath(__file__))
ffmpeg_path = os.path.join(project_dir, "ffmpeg_bin")

# Add it to the system PATH for this run only
if os.path.exists(ffmpeg_path):
    os.environ["PATH"] += os.pathsep + ffmpeg_path
else:
    print("WARNING: ffmpeg_bin folder not found. Audio analysis might fail.")
# --- FFMPEG PORTABLE SETUP (End) ---

import time
from collections import deque

from flask import Flask, render_template, request, redirect, url_for, Response
from flask import request, jsonify
from PIL import Image
import numpy as np
import base64
import cv2
from PIL import Image
import torch
from torchvision import transforms
import librosa

# Local imports
from config import DEVICE, INPUT_SIZE, IDX2EMOTION, TRAINED_MODEL_PATH, AUDIO_SAMPLE_RATE
from models.emotion_model import load_model, GradCAM
from mental_health_insights import get_wellbeing_summary

# Multimodal (new pipeline)
from multimodal.text_emotion import classify_text
from multimodal.audio_emotion import classify_audio

# Flask App
app = Flask(__name__)

# ----------------------------------------------------------
# Load Models
# ----------------------------------------------------------

# Facial Emotion Model
model = load_model(TRAINED_MODEL_PATH, DEVICE)
gradcam = GradCAM(model)


# ----------------------------------------------------------
# Preprocessing & Face Detector
# ----------------------------------------------------------
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
preprocess = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

face_cascade = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
)

# Emotion memory (for wellbeing)
EMOTION_HISTORY = deque(maxlen=50)

# ----------------------------------------------------------
# Home
# ----------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

# ----------------------------------------------------------
# IMAGE UPLOAD
# ----------------------------------------------------------
@app.route("/upload")
def upload_page():
    return render_template("upload.html")


@app.route("/predict_image", methods=["POST"])
def predict_image_route():
    if "image" not in request.files:
        return redirect(url_for("upload_page"))

    file = request.files["image"]
    if file.filename == "":
        return redirect(url_for("upload_page"))

    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Face detection
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if len(faces) == 0:
        face_roi = img
    else:
        x, y, w, h = faces[0]
        face_roi_bgr = img_bgr[y:y + h, x:x + w]
        face_roi = Image.fromarray(cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2RGB))

    # Preprocess
    input_tensor = preprocess(face_roi).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_label = IDX2EMOTION[pred_idx]

    # GradCAM
    cam = gradcam.generate(input_tensor)
    cam_resized = cv2.resize(cam, (face_roi.size[0], face_roi.size[1]))
    heatmap = (cam_resized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    face_roi_bgr = cv2.cvtColor(np.array(face_roi), cv2.COLOR_RGB2BGR)
    overlay = (0.4 * heatmap + 0.6 * face_roi_bgr).astype(np.uint8)

    # Save outputs
    results_dir = os.path.join("static", "results")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = int(time.time())
    orig_path = os.path.join(results_dir, f"uploaded_{timestamp}.jpg")
    gradcam_path = os.path.join(results_dir, f"gradcam_{timestamp}.jpg")

    cv2.imwrite(orig_path, img_bgr)
    cv2.imwrite(gradcam_path, overlay)

    EMOTION_HISTORY.append(pred_label)
    wellbeing = get_wellbeing_summary(list(EMOTION_HISTORY))

    prob_dict = {IDX2EMOTION[i]: float(probs[i]) for i in range(len(probs))}

    return render_template(
        "result.html",
        predicted_label=pred_label,
        probabilities=prob_dict,
        orig_image=url_for("static", filename=f"results/uploaded_{timestamp}.jpg"),
        gradcam_image=url_for("static", filename=f"results/gradcam_{timestamp}.jpg"),
        wellbeing=wellbeing,
    )

# ----------------------------------------------------------
# TEXT EMOTION
# ----------------------------------------------------------
@app.route("/text", methods=["GET"])
def text_page():
    return render_template("text.html")


@app.route("/predict_text", methods=["POST"])
def predict_text_route():
    user_text = request.form.get("user_text", "").strip()
    if not user_text:
        return redirect(url_for("text_page"))

    mode = request.form.get("mode", "accurate")
    allow_hf = (mode == "accurate")

    try:
        result = classify_text(user_text, allow_hf_fallback=allow_hf)
    except RuntimeError as e:
        return render_template("text_unavailable.html", error_message=str(e))

    predicted_label = result["predicted_label"]
    probabilities = result["scores"]

    EMOTION_HISTORY.append(predicted_label)
    wellbeing = get_wellbeing_summary(list(EMOTION_HISTORY))

    return render_template(
        "result_text.html",
        user_text=user_text,
        predicted_label=predicted_label,
        probabilities=probabilities,
        wellbeing=wellbeing,
    )

# ----------------------------------------------------------
# AUDIO EMOTION
# ----------------------------------------------------------
@app.route("/audio", methods=["GET"])
def audio_page():
    return render_template("audio.html")


@app.route("/predict_audio", methods=["POST"])
def predict_audio_route():
    if "audio" not in request.files:
        return redirect(url_for("audio_page"))

    file = request.files["audio"]
    if file.filename == "":
        return redirect(url_for("audio_page"))

    # Save file
    audio_dir = os.path.join("static", "audio")
    os.makedirs(audio_dir, exist_ok=True)
    save_path = os.path.join(audio_dir, file.filename)
    file.save(save_path)

    # Read mode selected by user: fast / accurate
    mode = request.form.get("mode", "fast")

    try:
        result = classify_audio(save_path, mode=mode)
    except RuntimeError as e:
        return render_template("audio_unavailable.html", error_message=str(e))

    pred_label = result["predicted_label"]
    prob_dict = result["scores"]

    EMOTION_HISTORY.append(pred_label)
    wellbeing = get_wellbeing_summary(list(EMOTION_HISTORY))

    return render_template(
        "result_audio.html",
        audio_file=url_for("static", filename=f"audio/{file.filename}"),
        predicted_label=pred_label,
        probabilities=prob_dict,
        wellbeing=wellbeing,
    )


# ----------------------------------------------------------
# LIVE MODE (Webcam)
# ----------------------------------------------------------
@app.route("/live")
def live_page():
    return render_template("live.html")


# ----------------------------------------------------------
# NEW ROUTE: Process Visitor's Camera Frame
# ----------------------------------------------------------
@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No image data'}), 400

        # Decode Image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Face Detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # --- SPEED OPTIMIZATIONS ---
        # scaleFactor=1.2 (Faster than 1.1, slightly less accurate but good for video)
        # minNeighbors=5 (Standard setting, good balance)
        # minSize=(40, 40) (Catches slightly smaller faces)
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.2, 
            minNeighbors=5, 
            minSize=(40, 40)
        )

        response_data = {'emotion': 'No Face', 'face_coords': []}

        if len(faces) > 0:
            # Get the largest face
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            
            # --- PREDICTION LOGIC ---
            face_bgr = frame[y:y + h, x:x + w]
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            
            input_tensor = preprocess(face_pil).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            pred_idx = int(np.argmax(probs))
            pred_label = IDX2EMOTION[pred_idx]
            
            EMOTION_HISTORY.append(pred_label)

            response_data = {
                'emotion': pred_label,
                'face_coords': [int(x), int(y), int(w), int(h)]
            }

        return jsonify(response_data)

    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500
# ----------------------------------------------------------
# Run App
# ----------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
