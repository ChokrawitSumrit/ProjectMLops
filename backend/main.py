from pathlib import Path
import traceback
import joblib
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# =========================
# Path config
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "job_category_model.pkl"
LABEL_PATH = BASE_DIR / "model" / "label_encoder.pkl"


# =========================
# Load model and label encoder
# =========================
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not LABEL_PATH.exists():
    raise FileNotFoundError(f"Label encoder file not found: {LABEL_PATH}")

print(f"[DEBUG] BASE_DIR   = {BASE_DIR}")
print(f"[DEBUG] MODEL_PATH = {MODEL_PATH}")
print(f"[DEBUG] LABEL_PATH = {LABEL_PATH}")
print(f"[DEBUG] MODEL exists = {MODEL_PATH.exists()} size = {MODEL_PATH.stat().st_size} bytes")
print(f"[DEBUG] LABEL exists = {LABEL_PATH.exists()} size = {LABEL_PATH.stat().st_size} bytes")

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_PATH)

print(f"[DEBUG] Loaded model type = {type(model)}")
print(f"[DEBUG] Loaded label encoder type = {type(label_encoder)}")

# ทดสอบ model ตั้งแต่ startup เลย
try:
    _startup_test_text = "workplace remote location bangkok thailand department analytics job type full_time business analyst reporting dashboard sql excel"
    _startup_pred = model.predict([_startup_test_text])
    print(f"[DEBUG] Startup test predict OK -> {_startup_pred}")
except Exception as e:
    print("[DEBUG] Startup test predict FAILED")
    print(traceback.format_exc())
    raise


# =========================
# FastAPI app
# =========================
app = FastAPI(title="AI Job Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Request / Response schema
# =========================
class JobRequest(BaseModel):
    workplace: str
    location: str
    department: str
    job_type: str


class TopPrediction(BaseModel):
    job: str
    score: float


class JobResponse(BaseModel):
    recommended_job: str
    raw_model_prediction: str
    rule_applied: bool
    applied_rule: str
    reason: str
    model_input_text: str
    top_predictions: list[TopPrediction]


# =========================
# Utility functions
# =========================
def normalize_text(text: str) -> str:
    return str(text).strip().lower()


def build_input_text(workplace: str, location: str, department: str, job_type: str) -> str:
    workplace = normalize_text(workplace)
    location = normalize_text(location)
    department = normalize_text(department)
    job_type = normalize_text(job_type)

    extra_terms = []

    if "analytics" in department:
        extra_terms.extend([
            "business analyst",
            "business analysis",
            "reporting",
            "dashboard",
            "stakeholder communication",
            "excel",
            "sql",
            "data interpretation"
        ])
    elif any(k in department for k in ["analysis", "analyst", "bi", "reporting"]):
        extra_terms.extend([
            "business analyst",
            "business analysis",
            "dashboard",
            "reporting",
            "excel",
            "sql",
            "stakeholder communication"
        ])
    elif "cloud" in department:
        extra_terms.extend([
            "cloud engineer",
            "aws",
            "azure",
            "gcp",
            "docker",
            "kubernetes",
            "linux",
            "terraform",
            "networking",
            "devops"
        ])
    elif any(k in department for k in ["data", "science", "ml", "ai"]):
        extra_terms.extend([
            "data scientist",
            "python",
            "sql",
            "pandas",
            "numpy",
            "machine learning",
            "statistics",
            "data visualization",
            "predictive modeling"
        ])
    elif "hr" in department or "human resource" in department:
        extra_terms.extend([
            "human resources",
            "recruitment",
            "onboarding",
            "payroll",
            "employee relations",
            "talent acquisition",
            "people management"
        ])
    elif any(k in department for k in ["software", "developer", "development", "engineering"]):
        extra_terms.extend([
            "software developer",
            "python",
            "java",
            "javascript",
            "sql",
            "git",
            "api",
            "backend",
            "frontend"
        ])
    elif any(k in department for k in ["ui", "ux", "design", "product design"]):
        extra_terms.extend([
            "ui ux designer",
            "figma",
            "wireframe",
            "prototype",
            "user research",
            "usability",
            "design system",
            "visual design"
        ])

    input_text = f"""
    workplace {workplace}
    location {location}
    department {department}
    job type {job_type}
    {' '.join(extra_terms)}
    """

    return " ".join(input_text.split())


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()


def get_top_predictions(model, label_encoder, input_text: str, top_k: int = 3):
    if hasattr(model, "decision_function"):
        scores = model.decision_function([input_text])

        if len(scores.shape) == 2:
            scores = scores[0]
            probs = softmax(scores)
            top_idx = np.argsort(probs)[::-1][:top_k]

            results = []
            for idx in top_idx:
                label = label_encoder.inverse_transform([idx])[0]
                results.append({
                    "job": label,
                    "score": round(float(probs[idx]), 4)
                })
            return results

        else:
            score = float(scores[0])
            probs = softmax(np.array([-score, score]))
            class_ids = [0, 1]
            top_idx = np.argsort(probs)[::-1][:top_k]

            results = []
            for idx in top_idx:
                label = label_encoder.inverse_transform([class_ids[idx]])[0]
                results.append({
                    "job": label,
                    "score": round(float(probs[idx]), 4)
                })
            return results

    elif hasattr(model, "predict_proba"):
        probs = model.predict_proba([input_text])[0]
        top_idx = np.argsort(probs)[::-1][:top_k]

        results = []
        for idx in top_idx:
            label = label_encoder.inverse_transform([idx])[0]
            results.append({
                "job": label,
                "score": round(float(probs[idx]), 4)
            })
        return results

    else:
        pred = model.predict([input_text])[0]
        try:
            label = label_encoder.inverse_transform([pred])[0]
        except Exception:
            label = str(pred)

        return [{
            "job": label,
            "score": 1.0
        }]


def apply_business_rules(workplace, location, department, job_type, predicted_label):
    department = normalize_text(department)

    if "analytics" in department:
        return True, "analytics_department_override", "Business Analyst"
    if any(k in department for k in ["analysis", "analyst", "bi", "reporting", "dashboard"]):
        return True, "analyst_keyword_override", "Business Analyst"
    if "cloud" in department:
        return True, "cloud_department_override", "Cloud"
    if department == "hr" or "human resource" in department:
        return True, "hr_department_override", "HR"
    if any(k in department for k in ["ui", "ux", "design"]):
        return True, "design_department_override", "UI/UX"
    if any(k in department for k in ["software", "developer", "development", "engineering"]):
        return True, "software_department_override", "Software Developer"
    if any(k in department for k in ["data science", "machine learning", "ml", "ai", "data scientist"]):
        return True, "data_science_department_override", "Data Scientist"

    return False, "", predicted_label


def build_reason(workplace, location, department, job_type, recommended_job, rule_applied, applied_rule, raw_model_prediction):
    if rule_applied:
        return (
            f"ระบบแนะนำตำแหน่ง {recommended_job} โดยใช้กฎธุรกิจเพิ่มเติม ({applied_rule}) "
            f"เพราะข้อมูลที่กรอกสอดคล้องกับ department = {department}, workplace = {workplace}, "
            f"location = {location}, job_type = {job_type} แม้โมเดลดิบจะทำนายเป็น {raw_model_prediction}"
        )

    return (
        f"ระบบแนะนำตำแหน่ง {recommended_job} เพราะข้อมูลที่กรอกสอดคล้องกับ "
        f"department = {department}, workplace = {workplace}, location = {location} และ job_type = {job_type}"
    )


def predict_job(workplace: str, location: str, department: str, job_type: str):
    input_text = build_input_text(workplace, location, department, job_type)
    top_predictions = get_top_predictions(model, label_encoder, input_text, top_k=3)
    raw_model_prediction = top_predictions[0]["job"]

    rule_applied, applied_rule, final_prediction = apply_business_rules(
        workplace=workplace,
        location=location,
        department=department,
        job_type=job_type,
        predicted_label=raw_model_prediction
    )

    existing_jobs = [item["job"] for item in top_predictions]
    if final_prediction not in existing_jobs:
        top_predictions = [{"job": final_prediction, "score": 0.9999}] + top_predictions[:2]
    else:
        reordered = []
        first_item = None
        for item in top_predictions:
            if item["job"] == final_prediction and first_item is None:
                first_item = item
            else:
                reordered.append(item)
        if first_item is not None:
            top_predictions = [first_item] + reordered

    reason = build_reason(
        workplace=workplace,
        location=location,
        department=department,
        job_type=job_type,
        recommended_job=final_prediction,
        rule_applied=rule_applied,
        applied_rule=applied_rule,
        raw_model_prediction=raw_model_prediction
    )

    return {
        "recommended_job": final_prediction,
        "raw_model_prediction": raw_model_prediction,
        "rule_applied": rule_applied,
        "applied_rule": applied_rule,
        "reason": reason,
        "model_input_text": input_text,
        "top_predictions": top_predictions
    }


@app.get("/")
def root():
    return {"message": "AI Job Recommendation API is running"}


@app.post("/predict", response_model=JobResponse)
def predict(request: JobRequest):
    try:
        return predict_job(
            workplace=request.workplace,
            location=request.location,
            department=request.department,
            job_type=request.job_type
        )
    except Exception as e:
        print("[DEBUG] /predict FAILED")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))