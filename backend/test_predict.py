import re
import joblib
from pathlib import Path

# ==========================
# หา root ของ project
# ==========================
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "job_category_model.pkl"
ENCODER_PATH = BASE_DIR / "model" / "label_encoder.pkl"

print("MODEL PATH:", MODEL_PATH)

# ==========================
# LOAD MODEL
# ==========================
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)


def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s\-/&,+_]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==========================
# SAMPLE INPUT
# ==========================
sample_text = ("workplace remote location bangkok department analytics job_type full_time")

sample_text = clean_text(sample_text)

# ==========================
# PREDICT
# ==========================
pred_num = model.predict([sample_text])[0]
pred_label = label_encoder.inverse_transform([pred_num])[0]

print("Input Text:", sample_text)
print("Predicted Job:", pred_label)