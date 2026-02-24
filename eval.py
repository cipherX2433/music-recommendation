# ============================================
# Evaluation for Emotion-Based Music System
# Face | Text | Late Fusion | Neuro-Symbolic
# ============================================

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# -------------------------------
# EMOTION SETUP
# -------------------------------
EMOTIONS = ["happy", "sad", "angry", "fear", "neutral"]
EMO2ID = {e: i for i, e in enumerate(EMOTIONS)}

ALPHA = 0.6
BETA = 0.4

# -------------------------------
# MANUAL TEST CASES
# (Paste probabilities from your app here)
# -------------------------------
TEST_CASES = [
    {
        "face_probs": [0.134, 0.308, 0.335, 0.119, 0.103],
        "text_probs": [0.015, 0.979, 0.002, 0.002, 0.003],
        "true": "sad"
    },
    {
        "face_probs": [0.10, 0.65, 0.10, 0.05, 0.10],
        "text_probs": [0.05, 0.85, 0.05, 0.03, 0.02],
        "true": "sad"
    }
]

# -------------------------------
# FUSION
# -------------------------------
def late_fusion(face, text):
    return ALPHA * np.array(face) + BETA * np.array(text)

# -------------------------------
# NEURO-SYMBOLIC LOGIC
# -------------------------------
RULES = {
    ("sad", "neutral"): "sad",
    ("fear", "neutral"): "fear",
    ("happy", "sad"): "neutral"
}

def neuro_symbolic(face_em, text_em, fused_probs, text_probs):
    fused_em = EMOTIONS[np.argmax(fused_probs)]

    # 1️⃣ Trust text if highly confident
    if np.max(text_probs) >= 0.9:
        return text_em

    # 2️⃣ Trust fusion if confident
    if np.max(fused_probs) >= 0.5:
        return fused_em

    # 3️⃣ Otherwise apply symbolic rules
    return RULES.get((face_em, text_em), fused_em)

# -------------------------------
# EVALUATION LOOP
# -------------------------------
y_true = []
y_face = []
y_text = []
y_fusion = []
y_final = []

print("\n--- Individual Predictions ---")
for i, case in enumerate(TEST_CASES, 1):
    face_probs = np.array(case["face_probs"])
    text_probs = np.array(case["text_probs"])
    true_em = case["true"]

    face_em = EMOTIONS[np.argmax(face_probs)]
    text_em = EMOTIONS[np.argmax(text_probs)]
    fused_probs = late_fusion(face_probs, text_probs)
    fused_em = EMOTIONS[np.argmax(fused_probs)]
    final_em = neuro_symbolic(face_em, text_em, fused_probs, text_probs)

    print(f"\nSample {i}")
    print("True Label :", true_em.upper())
    print("Face Pred  :", face_em.upper())
    print("Text Pred  :", text_em.upper())
    print("Final Pred :", final_em.upper())

    y_true.append(EMO2ID[true_em])
    y_face.append(EMO2ID[face_em])
    y_text.append(EMO2ID[text_em])
    y_fusion.append(EMO2ID[fused_em])
    y_final.append(EMO2ID[final_em])

# -------------------------------
# METRICS
# -------------------------------
def evaluate(name, y_pred):
    print(f"\n===== {name} =====")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("F1-score :", f1_score(y_true, y_pred, average="weighted"))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

evaluate("Face Only", y_face)
evaluate("Text Only", y_text)
evaluate("Late Fusion", y_fusion)
evaluate("Fusion + Neuro-Symbolic", y_final)

# -------------------------------
# SAFE CLASSIFICATION REPORT
# -------------------------------
print("\nClassification Report (Final Model):")
print(
    classification_report(
        y_true,
        y_final,
        labels=list(range(len(EMOTIONS))),
        target_names=EMOTIONS,
        zero_division=0
    )
)
