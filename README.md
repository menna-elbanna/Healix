# Healix — The Educational Assistant

A multimodal deep learning pipeline that takes an image, generates an English caption, translates it to Arabic, and produces questions from it.

---

## Team & Pipeline

| Member | Task | Dataset |
|--------|------|---------|
| Menna | EfficientNetB0 Image Encoder | Flickr8k |
| Malak | LSTM Caption Decoder (English) | Flickr8k |
| Pola | Seq2Seq English → Arabic Translator | Opus100 |
| Alber | LSTM Question Generator | SQuAD |

**Full pipeline:**
```
Image → [Menna: EfficientNetB0] → features → [Malak: LSTM Decoder] → English caption
→ [Pola: Seq2Seq] → Arabic caption → [Alber: LSTM] → Questions
```

---
> Notebooks are the deliverables. Datasets, models, and feature files are NOT tracked in git (see .gitignore).
---

## Datasets (not in repo — download separately)

| Dataset | Used by | Download |
|---------|---------|----------|
| Flickr8k (images + captions) | Menna, Malak | [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k) |
| Opus100 (en-ar parallel corpus) | Pola | [Hugging Face](https://huggingface.co/datasets/opus100) |
| SQuAD | Alber | [Rajpurkar et al.](https://rajpurkar.github.io/SQuAD-explorer/) |

Place Flickr8k images at: `data/Images/` (not inside any subfolder).

---

## .gitignore Rules

Do NOT commit:
- Any dataset files
- `.keras` model files
- `.npy` feature files
- `outputs/` folder contents

---

## For Malak — LSTM Decoder

Menna's encoder produces a `(1280,)` feature vector per image using EfficientNetB0 pretrained on ImageNet.

**How to load the features:**
```python
import numpy as np

features = np.load("path/to/all_features.npy", allow_pickle=True).item()
# features is a dict: { "image_filename.jpg": np.array of shape (1280,) }

# example
vec = features["667626_18933d713e.jpg"]
print(vec.shape)  # (1280,)
```

**What you'll receive from Menna (via Google Drive):**
- `all_features.npy` — dict of all 8091 image feature vectors
- `efficientnetb0_encoder.keras` — the saved encoder model (optional, if you need to reload it)

**Technical details:**
- Framework: TensorFlow / Keras (standalone `from keras import ...`)
- Input: raw pixel images resized to (224, 224, 3)
- Output: (1280,) float32 vectors, one per image
- Preprocessing: `preprocess_input` from `keras.applications.efficientnet` — not `/255`
- Total images: 8091

---

## For Pola — Seq2Seq Translator

You receive English captions from Malak's LSTM decoder. Your job is translating them to Arabic using the Opus100 en-ar corpus.

---

## For Alber — Question Generator

You receive Arabic captions from Pola's translator. Your job is generating questions from them using an LSTM trained on SQuAD.

---

## Framework & Environment

- Python 3.x
- TensorFlow >= 2.11
- Keras (standalone imports — `from keras import ...`, NOT `from tensorflow.keras import ...`)
- IDE: VS Code with Jupyter notebooks
- OS: Windows (CPU only — GPU warnings from TF can be ignored)

**Install dependencies:**
```
pip install tensorflow keras numpy matplotlib
```

---

## Notes

- Pylance may show red underlines on some keras imports — these are warnings, not errors. The code runs fine.
- TensorFlow on Windows >= 2.11 has no GPU support — the warning on startup is expected and harmless.
- EfficientNetB0 expects raw pixel values [0, 255] — always use `preprocess_input`, never `rescale=1./255`.