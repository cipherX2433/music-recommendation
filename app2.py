import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMOTIONS = ["happy", "sad", "angry", "fear", "neutral"]
NUM_EMOTIONS = len(EMOTIONS)

ALPHA = 0.6   # face weight
BETA  = 0.4   # text weight

st.set_page_config(page_title="Emotion Music Recommender", layout="centered")

@st.cache_data
def load_muse():
    return pd.read_csv("muse_v3.csv")

df = load_muse()


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
    else:
        df2 = df2[
            (df2.valence_tags.between(0.4, 0.6)) &
            (df2.arousal_tags.between(0.4, 0.6))
        ]

    if len(df2) == 0:
        df2 = df.sample(k)

    return df2.sample(min(k, len(df2)))[
        ["track", "artist", "genre", "lastfm_url"]
    ]

@st.cache_resource
def load_resnet():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
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


@st.cache_resource
def load_bert():
    tokenizer = AutoTokenizer.from_pretrained("nateraw/bert-base-uncased-emotion")
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


def late_fusion(face_probs, text_probs):
    return ALPHA * face_probs + BETA * text_probs

# NSI

RULES = {
    ("sad", "neutral"): "sad",
    ("fear", "neutral"): "fear",
    ("happy", "sad"): "neutral"
}

def apply_neuro_symbolic(face_em, text_em, fused_probs, text_probs):
    fused_em = EMOTIONS[np.argmax(fused_probs)]
    fused_conf = np.max(fused_probs)
    text_conf = np.max(text_probs)

    if text_conf >= 0.9:
        return text_em

    if fused_conf >= 0.5:
        return fused_em


    return RULES.get((face_em, text_em), fused_em)


def show_probs(title, probs):
    st.markdown(f"### {title}")
    for emo, p in zip(EMOTIONS, probs):
        st.write(f"**{emo.capitalize()}** : {p:.3f}")

#ui
st.title("Emotion-Based Music Recommendation")
st.write("**ResNet + BERT + Late Fusion + Neuro-Symbolic AI**")

st.subheader("Capture Your Face")
image_file = st.camera_input("Take a picture")

st.subheader("Describe Your Feelings")
user_text = st.text_area("Example: I feel lonely and tired today")

if st.button("Predict Emotion & Recommend Music"):
    if image_file is None and not user_text.strip():
        st.warning("Please provide face image or text input")
        st.stop()

    # Face
    if image_file:
        image = Image.open(image_file).convert("RGB")
        face_probs = face_emotion_predict(image)
        face_em = EMOTIONS[np.argmax(face_probs)]
        st.success(f"Face Emotion: **{face_em.upper()}**")
        show_probs("Face Emotion Probabilities", face_probs)
    else:
        face_probs = np.zeros(NUM_EMOTIONS)
        face_em = "neutral"

    # Text
    if user_text.strip():
        text_probs = text_emotion_predict(user_text)
        text_em = EMOTIONS[np.argmax(text_probs)]
        st.success(f"Text Emotion: **{text_em.upper()}**")
        show_probs("Text Emotion Probabilities", text_probs)
    else:
        text_probs = np.zeros(NUM_EMOTIONS)
        text_em = "neutral"

    # Fusion
    fused_probs = late_fusion(face_probs, text_probs)
    show_probs("Fused Emotion Probabilities", fused_probs)

    # Neuro-symbolic decision
    final_em = apply_neuro_symbolic(
        face_em, text_em, fused_probs, text_probs
    )

    st.subheader(f"Final Emotion: **{final_em.upper()}**")

    # Songs
    st.subheader("Recommended Songs")
    songs = recommend_music(final_em)

    for _, row in songs.iterrows():
        st.markdown(
            f"""
            **{row['track']}**  
            **Artist:** {row['artist']}  
            **Genre:** {row['genre']}  
            [Listen on Last.fm]({row['lastfm_url']})
            ---
            """
        )