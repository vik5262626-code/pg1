import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="CTC OCR Demo",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# Character Map (EDIT IF NEEDED)
# index 0 is reserved for CTC blank
# -----------------------------
CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
idx2char = {i + 1: c for i, c in enumerate(CHARSET)}
idx2char[0] = ""  # CTC blank

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model = torch.load("best_ctc_ocr.pt", map_location="cpu")
    model.eval()
    return model

model = load_model()

# -----------------------------
# Image Preprocessing
# -----------------------------
transform = T.Compose([
    T.Grayscale(num_output_channels=1),
    T.Resize((32, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])

# -----------------------------
# CTC Decoding
# -----------------------------
def greedy_decode(logits):
    probs = F.softmax(logits, dim=2)
    max_probs, indices = probs.max(dim=2)

    decoded = []
    confidences = []

    prev = -1
    word_conf = []

    for idx, conf in zip(indices[0], max_probs[0]):
        idx = idx.item()
        conf = conf.item()
        if idx != prev and idx != 0:
            decoded.append(idx2char[idx])
            word_conf.append(conf)
        prev = idx

    text = "".join(decoded)
    confidence = float(np.mean(word_conf)) if word_conf else 0.0
    return text, confidence

# -----------------------------
# Streamlit UI
# -----------------------------
st.markdown("## 🧠 CTC OCR Text Recognition")
st.markdown("Supports batch OCR, greedy decoding, and word-level confidence.")
st.divider()

with st.sidebar:
    st.header("⚙️ Settings")
    decoding_mode = st.selectbox("Decoding", ["Greedy", "Beam (placeholder)"])
    st.markdown("Model: **best_ctc_ocr.pt**")
    st.markdown(f"Charset size: **{len(CHARSET)}**")

# -----------------------------
# Batch Upload
# -----------------------------
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
                        logits = model(img)  # (T, B, C) or (B, T, C)
                        if logits.dim() == 3 and logits.shape[0] != 1:
                            logits = logits.permute(1, 0, 2)

                        text, conf = greedy_decode(logits)

                st.success("OCR completed")
                st.text_area("Extracted Text", text, height=120)
                st.metric("Word-level Confidence", f"{conf * 100:.2f}%")

else:
    st.info("Upload one or more images to begin OCR.")

st.divider()
st.caption("CTC OCR • PyTorch • Streamlit")
