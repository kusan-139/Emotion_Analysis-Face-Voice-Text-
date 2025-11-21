from transformers import AutoTokenizer, AutoModelForSequenceClassification

HF_TEXT_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
save_dir = "models/hf_text_model"

tokenizer = AutoTokenizer.from_pretrained(HF_TEXT_MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(HF_TEXT_MODEL_NAME)

tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)
