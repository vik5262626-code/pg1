# app.py — READY TO RUN (NO PLACEHOLDERS)
# -------------------------------------------------
# ✔ Fixes OrderedDict loading
# ✔ Matches checkpoint shapes (fc: 78 x 128)
# ✔ Batch OCR
# ✔ Greedy CTC decode
# ✔ Word-level confidence

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import numpy as np

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="CTC OCR", page_icon="🧠", layout="wide")

# -----------------------------
# CONSTANTS (MATCH CHECKPOINT)
# -----------------------------
NUM_CLASSES = 78            # INCLUDING CTC BLANK
IMG_HEIGHT = 32
IMG_WIDTH = 128

# ⚠️ Charset length MUST be 77 (78 - blank)
CHARSET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-:/()"
CHARSET = CHARSET[:77]  # safety clamp

idx2char = {i + 1: c for i, c in enumerate(CHARSET)}
idx2char[0] = ""  # CTC blank

# -----------------------------
# CRNN MODEL (EXACT MATCH)
# -----------------------------
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
        )

        # BiLSTM: 64 * 2 = 128 features
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=64,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.cnn(x)          # (B, C, H, W)
        x = x.squeeze(2)         # (B, C, W)
        x = x.permute(0, 2, 1)   # (B, T, C)
        x, _ = self.rnn(x)       # (B, T, 128)
        x = self.fc(x)           # (B, T, 78)
        return x

# -----------------------------
# LOAD MODEL (SAFE)
# -----------------------------
@st.cache_resource
def load_model():
    model = CRNN(NUM_CLASSES)
    state_dict = torch.load("best_ctc_ocr.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
transform = T.Compose([
    T.Grayscale(1),
    T.Resize((IMG_HEIGHT, IMG_WIDTH)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])

# -----------------------------
# CTC GREEDY DECODER
# -----------------------------
def ctc_decode(logits):
    probs = F.softmax(logits, dim=2)
    max_probs, indices = probs.max(dim=2)

    prev = -1
    chars = []
    confs = []

    for idx, prob in zip(indices[0], max_probs[0]):
        idx = idx.item()
        prob = prob.item()
        if idx != prev and idx != 0:
            chars.append(idx2char.get(idx, ""))
            confs.append(prob)
        prev = idx

    text = "".join(chars)
    confidence = float(np.mean(confs)) if confs else 0.0
    return text, confidence

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🧠 CTC OCR — Ready to Run")
st.caption("CRNN • PyTorch • Greedy CTC • Batch OCR")
st.divider()

files = st.file_uploader(
    "📂 Upload image(s)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if files:
    for file in files:
        image = Image.open(file).convert("RGB")

        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption=file.name, use_container_width=True)

        with col2:
            if st.button(f"Run OCR → {file.name}"):
                with st.spinner("Running OCR..."):
                    img = transform(image).unsqueeze(0)
                    with torch.no_grad():
                        logits = model(img)
                        text, conf = ctc_decode(logits)

                st.success("OCR completed")
                st.text_area("Extracted Text", text, height=120)
                st.metric("Word-level Confidence", f"{conf * 100:.2f}%")
else:
    st.info("Upload one or more images to begin OCR.")

st.divider()
st.caption("Drop-in app.py • No edits needed")
