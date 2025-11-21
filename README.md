# Facial Emotion Recognition with Wellbeing-Aware Insights  
*(FER2013 + RAF-DB Combined Dataset)*

## Abstract

This project implements a **production-ready facial emotion recognition system**
using **deep learning (ResNet-18, PyTorch)** and a **combined dataset**
from **FER2013** and **RAF-DB (basic subset)**.

On top of emotion recognition, it provides **non-clinical wellbeing indicators**
(positivity, negativity, neutrality scores and a simple concern level) based on
recent visible emotional patterns.

> ⚠️ **Strong Disclaimer**  
> This project does **NOT** diagnose any mental health or medical condition.  
> All outputs are informational only and must **never** be used as a substitute
> for professional advice, diagnosis, or treatment.

---

## 1. Emotion Recognition Overview

Facial emotion recognition aims to classify expressions such as:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

This project uses **transfer learning** from ImageNet-pretrained ResNet-18,
fine-tuned on a combined facial dataset, to provide robust recognition.

---

## 2. Combined Dataset: FER2013 + RAF-DB

### FER2013

- ~36k grayscale 48×48 images
- 7 emotion classes (0–6)
- Provided as `fer2013.csv` with columns: `emotion`, `pixels`, `Usage`

### RAF-DB (Real-world Affective Faces Database)

- ~30k high-quality RGB images
- We use the **basic emotion subset** only (no compound emotions)
- Each image has a label in `list_patition_label.txt`

### Unified Label Mapping

All emotions are mapped to a consistent 7-class scheme:

| Index | Label    |
|-------|----------|
| 0     | Angry    |
| 1     | Disgust  |
| 2     | Fear     |
| 3     | Happy    |
| 4     | Sad      |
| 5     | Surprise |
| 6     | Neutral  |

All images are resized and normalized to **224×224 RGB**.

Final combined dataset structure:

```text
dataset_combined/
  train/
    angry/
    disgust/
    fear/
    happy/
    sad/
    surprise/
    neutral/
  val/
    angry/
    ...
  test/
    angry/
    ...
