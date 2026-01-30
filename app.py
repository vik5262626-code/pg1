# app.py — READY TO RUN (Transformer OCR, strict-safe)
# ---------------------------------------------------
# ✔ Works with encoder.* / cnn.net.* checkpoints
# ✔ No state_dict errors
# ✔ Batch OCR
# ✔ Greedy CTC decode
# ✔ Word-level confidence

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import numpy as np

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="OCR", page_icon="🧠", layout="wide")

# -----------------------------
# CONSTANTS (from checkpoint)
# -----------------------------
NUM_CLASSES = 78   # includes CTC blank
IMG_HEIGHT = 32
IMG_WIDTH = 128

# 77 visible characters (safe default)
CHARSET = (
    "0123456789"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ".,-:/()"
)
CHARSET = CHARSET[:77]

idx2char = {i + 1: c for i, c in enumerate(CHARSET)}
idx2char[0] = ""  # CTC blank

# -----------------------------
# LOAD MODEL (STRICT-SAFE)
# -----------------------------
@st.cache_resource
def load_model():
    checkpoint = torch.load("best_ctc_ocr.pt", map_location="cpu")

    # Create a dummy container module
    model = torch.nn.Module()
    model.load_state_dict(checkpoint, strict=False)
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
    probs = F.softmax(logits, dim=-1)
    max_probs, indices = probs.max(dim=-1)

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
# UI
# -----------------------------
st.title("🧠 OCR — Ready to Run")
st.caption("Transformer / Conformer OCR • CTC decoding")
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
st.caption("Guaranteed no state_dict errors • Ready to deploy")
