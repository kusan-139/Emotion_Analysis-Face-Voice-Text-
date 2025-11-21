import os
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

model_name = "superb/hubert-large-superb-er"
save_dir = "models/hf_audio_model"

os.makedirs(save_dir, exist_ok=True)

extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(model_name)

extractor.save_pretrained(save_dir)
model.save_pretrained(save_dir)

print(f"Saved pretrained model to {save_dir}")
