# ============================================
# Emotion Based Music Prediction - Streamlit
# ResNet + BERT + Late Fusion + Neuro-Symbolic
# ============================================

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image

# -------------------------------
# CONFIG
# -------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMOTIONS = ["happy", "sad", "angry", "fear", "neutral"]
NUM_EMOTIONS = len(EMOTIONS)

ALPHA = 0.6  # image weight
BETA = 0.4   # text weight

st.set_page_config(page_title="Emotion Music Recommender", layout="centered")

# -------------------------------
# LOAD MUSE DATASET
# -------------------------------
@st.cache_data
def load_muse():
    return pd.read_csv("muse_v3.csv")

df = load_muse()

# -------------------------------
# MUSIC RECOMMENDATION (VALENCE–AROUSAL)
# -------------------------------
def recommend_music(emotion, k=5):
    df2 = df.dropna(subset=["valence_tags", "arousal_tags"])

    if emotion == "happy":
        df2 = df2[(df2.valence_tags > 0.6) & (df2.arousal_tags > 0.5)]

    elif emotion == "sad":
        df2 = df2[(df2.valence_tags < 0.4) & (df2.arousal_tags < 0.4)]

    elif emotion == "angry":
        df2 = df2[(df2.valence_tags < 0.4) & (df2.arousal_tags > 0.7)]

    elif emotion == "fear":
        df2 = df2[(df2.valence_tags < 0.45) & (df2.arousal_tags > 0.6)]

    else:  # neutral
        df2 = df2[
            (df2.valence_tags.between(0.4, 0.6)) &
            (df2.arousal_tags.between(0.4, 0.6))
        ]

    if len(df2) == 0:
        df2 = df.sample(k)

    return df2.sample(min(k, len(df2)))[
        ["track", "artist", "genre", "lastfm_url"]
    ]

# -------------------------------
# LOAD RESNET (FACE EMOTION)
# -------------------------------
@st.cache_resource
def load_resnet():
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_EMOTIONS)
    model.to(DEVICE).eval()
    return model

resnet = load_resnet()

face_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

def face_emotion_predict(pil_img):
    img = face_transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(resnet(img), dim=1)
    return probs.cpu().numpy()[0]

# -------------------------------
# LOAD BERT (TEXT EMOTION)
# -------------------------------
@st.cache_resource
def load_bert():
    tokenizer = AutoTokenizer.from_pretrained(
        "nateraw/bert-base-uncased-emotion"
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "nateraw/bert-base-uncased-emotion"
    ).to(DEVICE).eval()
    return tokenizer, model

tokenizer, bert = load_bert()
bert_labels = list(bert.config.id2label.values())

BERT_TO_SYSTEM = {
    "joy": "happy",
    "sadness": "sad",
    "anger": "angry",
    "fear": "fear",
    "neutral": "neutral",
    "love": "happy",
    "surprise": "neutral"
}

def text_emotion_predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        logits = bert(**inputs).logits

    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    system_vec = np.zeros(NUM_EMOTIONS)
    for i, label in enumerate(bert_labels):
        if label in BERT_TO_SYSTEM:
            idx = EMOTIONS.index(BERT_TO_SYSTEM[label])
            system_vec[idx] += probs[i]

    return system_vec / np.sum(system_vec)

# -------------------------------
# FUSION + NEURO-SYMBOLIC AI
# -------------------------------
def late_fusion(face_probs, text_probs):
    return ALPHA * face_probs + BETA * text_probs

RULES = {
    ("sad", "neutral"): "sad",
    ("angry", "happy"): "angry",
    ("fear", "neutral"): "fear",
    ("happy", "sad"): "neutral"
}

def apply_neuro_symbolic(face_em, text_em, fused_em):
    return RULES.get((face_em, text_em), fused_em)

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("Emotion-Based Music Recommendation")
st.write("**ResNet + BERT + Late Fusion + Neuro-Symbolic AI**")

# Camera
st.subheader("Capture Your Face")
image_file = st.camera_input("Take a picture")

# Text
st.subheader("Describe Your Feelings")
user_text = st.text_area(
    "Example: I feel lonely and tired today"
)

# Run
if st.button("Predict Emotion & Recommend Music"):
    if image_file is None and not user_text.strip():
        st.warning("Please provide face image or text input")
        st.stop()

    # Face emotion
    if image_file:
        image = Image.open(image_file).convert("RGB")
        face_probs = face_emotion_predict(image)
        face_em = EMOTIONS[np.argmax(face_probs)]
        st.success(f"Face Emotion: **{face_em}**")
    else:
        face_probs = np.zeros(NUM_EMOTIONS)
        face_em = "neutral"

    # Text emotion
    if user_text.strip():
        text_probs = text_emotion_predict(user_text)
        text_em = EMOTIONS[np.argmax(text_probs)]
        st.success(f"Text Emotion: **{text_em}**")
    else:
        text_probs = np.zeros(NUM_EMOTIONS)
        text_em = "neutral"

    # Fusion
    fused_probs = late_fusion(face_probs, text_probs)
    fused_em = EMOTIONS[np.argmax(fused_probs)]

    # Neuro-symbolic reasoning
    final_em = apply_neuro_symbolic(
        face_em, text_em, fused_em
    )

    st.subheader(f"Final Emotion: **{final_em.upper()}**")

    # Recommendations
    st.subheader("Recommended Songs")
    songs = recommend_music(final_em)

    for _, row in songs.iterrows():
        st.markdown(
            f"**{row['track']}** — *{row['artist']}*  \n"
            f"[Listen on Last.fm]({row['lastfm_url']})"
        )
