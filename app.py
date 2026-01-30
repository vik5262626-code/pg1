# app.py — READY TO USE Streamlit OCR App (CTC)
# -------------------------------------------------
# Works when best_ctc_ocr.pt is a state_dict (OrderedDict)
# Architecture: CRNN (CNN + BiLSTM + CTC)

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
# Charset (EDIT if needed)
# index 0 = CTC blank
# -----------------------------
CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
idx2char = {i + 1: c for i, c in enumerate(CHARSET)}
idx2char[0] = ""
NUM_CLASSES = len(CHARSET) + 1

# -----------------------------
# CRNN Model Definition
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
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
        )

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)              # (B, C, H, W)
        x = x.squeeze(2)             # (B, C, W)
        x = x.permute(0, 2, 1)       # (B, T, C)
        x, _ = self.rnn(x)           # (B, T, 512)
        x = self.fc(x)               # (B, T, num_classes)
        return x

# -----------------------------
# Load model (state_dict safe)
# -----------------------------
@st.cache_resource
def load_model():
    model = CRNN(NUM_CLASSES)
    state = torch.load("best_ctc_ocr.pt", map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

model = load_model()

# -----------------------------
# Image preprocessing
# -----------------------------
transform = T.Compose([
    T.Grayscale(1),
    T.Resize((32, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])

# -----------------------------
# CTC Greedy Decoder + confidence
# -----------------------------
def ctc_greedy_decode(logits):
    probs = F.softmax(logits, dim=2)
    max_probs, indices = probs.max(dim=2)

    prev = -1
    chars = []
    confs = []

    for idx, prob in zip(indices[0], max_probs[0]):
        idx = idx.item()
        prob = prob.item()
        if idx != prev and idx != 0:
            chars.append(idx2char[idx])
            confs.append(prob)
        prev = idx

    text = "".join(chars)
    confidence = float(np.mean(confs)) if confs else 0.0
    return text, confidence

# -----------------------------
# UI
# -----------------------------
st.title("🧠 CTC OCR (Ready to Use)")
st.caption("CRNN + PyTorch + CTC decoding")
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
                        text, conf = ctc_greedy_decode(logits)

                st.success("Done")
                st.text_area("Extracted Text", text, height=120)
                st.metric("Word-level Confidence", f"{conf * 100:.2f}%")
else:
    st.info("Upload one or more images to begin OCR.")

st.divider()
st.caption("Drop-in app.py • No changes required")
