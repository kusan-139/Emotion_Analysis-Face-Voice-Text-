# Multimodal Emotion & Wellbeing Analysis  
*(Face · Text · Voice — Deep Learning + HuggingFace)*

## 🌟 Overview

This project is a **full multimodal emotion‑analysis system** supporting:

### ✅ **1. Facial Emotion Recognition (PyTorch · ResNet‑18)**  
- Uses combined **FER2013 + RAF‑DB** dataset  
- Grad‑CAM visualization  
- Face detection (OpenCV Haar Cascade)  
- Produces emotion + wellbeing indicator  

### ✅ **2. Text Emotion Analysis (HuggingFace)**  
- Uses a **local text‑classification model** (DistilBERT fine‑tuned) &  **hf-local model**
- Works fully offline  
- Produces emotion + tone probabilities  
- Integrates into wellbeing scoring  

### ✅ **3. Voice Tone Emotion Analysis (Audio CNN or HF Model)**  
- Two modes:
  - **Fast Mode** → Local CNN classifier  
  - **Accurate Mode** → Offline HuggingFace Wav2Vec2 model  
- Returns emotion + prosody‑based emotional tone  

### 🎯 **Wellbeing Insight Engine**
Tracks recent predicted emotions (sliding window of 50 samples) and generates:
- Positivity trend  
- Negative spikes  
- Stability rating  
- Simple non‑clinical wellbeing indicator  

> ⚠️ **Disclaimer:**  
> This project does **NOT** diagnose mental health or medical conditions.  
> It provides informational insights only.

---

# 📌 Features

## 🔹 Facial Emotion Features
- ResNet‑18‑based classifier  
- Preprocessing (normalization, resizing)  
- Grad‑CAM heatmaps  
- Real‑time webcam mode (`/live`)

## 🔹 Text Emotion Features
- Local HuggingFace emotion model  
- Supports multiple emotions (happy, sad, anger, fear, neutral, etc.)  
- Lightweight inference  
- Can process any free‑text description  

---

## 🔹 Voice Emotion Features
### 🎙️ Fast Mode (Local CNN)
- MFCC‑based  
- Lightweight & fast  

### 🎙️ Accurate Mode (HF Wav2Vec2)
- Works offline  
- Better accuracy  
- Slower inference  

Example Output:
```
Emotion: angry
Probabilities: { angry: 0.74, sad: 0.10, neutral: 0.08, ... }
```

---

# 📁 Project Structure

```
facial-emotion-wellbeing/
│
├── app.py
├── config.py
├── mental_health_insights.py
├── merge_datasets.py
├── data/
|   ├── fer2013/ (Face)
|   ├── rafdb/ (Face)
|   ├── savee/ (Audio)
|   ├──    / (Text)
|
│
├── models/
│   ├── emotion_model.py
│   ├── hf_text_model
|   ├── hf_audio_model
|   ├── trained_model.pth
|   └── audio_model.pth
│
├── multimodal/
│   ├── text_emotion.py
|   ├── audio_emotion.py
│   └── audio_emotion.py
│
├── static/
│   ├── css/
│   ├── js/
│   ├── results/   ← stores GradCAM & original images
│   └── audio/     ← stores uploaded/recorded audio
|
├── inference/
|   ├── predict_singl.py
|
├── training/
|   ├── evaluate.py
|   ├── evaluate_audio.py
|   ├── evaluate_text.py
|   ├── train.py
|   ├── train_audio.py
|   ├── train_text.py
|   ├── utils.py    
|
├── templates/
│   ├── index.html
│   ├── upload.html
│   ├── text.html
│   ├── audio.html
│   ├── live.html
|   ├── text_unavailable.py
│   ├── result.html
│   ├── result_text.html
│   └── result_audio.html
│
└── README.md
```

---

# 🚀 Installation

```
git clone <repo-url>
cd 
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```
python app.py
```

App runs at:

```
http://127.0.0.1:5000
```

---

# 🧠 Wellbeing Indicator Logic

Based on last 50 emotions:

- Repeated **negative emotion spikes** → “Low stability”
- Balanced mix of emotions → “Neutral / Stable”
- Majority positive emotions → “Good wellbeing trend”
- Sudden shifts → “Volatile emotional pattern”

Returns:
```
{
  "wellbeing_indicator": "Medium Concern",
  "insight_text": "Recent patterns show elevated sadness and anger..."
}
```

---

# 📦 Datasets Used (Face Model)

### FER2013 + RAF‑DB (basic)
Both merged into 7 emotions:
```
angry, disgust, fear, happy, sad, surprise, neutral
```

Dataset folder structure:
```
dataset/
  train/
  val/
  test/
```

---

# 📝 License
MIT License 

---

# ❤️ Credits
- PyTorch  
- HuggingFace Transformers  
- OpenCV  
- FER2013 dataset  
- RAF‑DB dataset  

---

If you want, I can also generate:  
✔ Badges  
✔ Screenshots  
✔ Model architecture diagrams  
✔ API route documentation

