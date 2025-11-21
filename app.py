import os
import io
import time
from collections import deque

from flask import Flask, render_template, request, redirect, url_for, Response
import numpy as np
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


def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        frame_emotions = []

        for (x, y, w, h) in faces:
            face_bgr = frame[y:y + h, x:x + w]
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)

            input_tensor = preprocess(face_pil).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            pred_idx = int(np.argmax(probs))
            pred_label = IDX2EMOTION[pred_idx]

            frame_emotions.append(pred_label)
            EMOTION_HISTORY.append(pred_label)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                pred_label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        ret, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ----------------------------------------------------------
# Run App
# ----------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
