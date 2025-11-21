from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

model_name = "superb/hubert-large-superb-er"

# Downloads config + model weights into ~/.cache/huggingface/
extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(model_name)

print("Model & feature extractor downloaded successfully!")
