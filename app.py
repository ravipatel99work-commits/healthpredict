from flask import Flask, render_template, request,redirect,url_for, session, jsonify
import numpy as np
import pandas as pd
import joblib, json, os
from pathlib import Path


# --- NEW IMPORTS for graph ---
import matplotlib
matplotlib.use('Agg')   # Non-GUI backend for Flask
import matplotlib.pyplot as plt
import io, base64
from datetime import datetime
import sqlite3

# --- Gemini API (Chatbot) ---
import google.generativeai as genai
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

gemini_model = genai.GenerativeModel("gemini-2.5-flash")



def query_gemini(prompt):
    try:
        response = gemini_model.generate_content(
    f"""
You are a helpful medical assistant.

User symptoms: {prompt}

Respond strictly in this format:

Possible Conditions:
- point 1
- point 2

Precautions:
- point 1
- point 2

Lifestyle Recommendations:
- point 1
- point 2

When to See a Doctor:
- point 1

Disclaimer:
This is not medical advice. Consult a qualified doctor.

Do not use special characters. Keep it clean and structured.
"""
)
        return response.text
    except Exception as e:
        return f"⚠️ Error connecting to Gemini: {str(e)}"


app = Flask(__name__)
app.secret_key = "secret123"


DB_PATH = "medications.db"

# --- Create DB automatically if missing ---
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS medications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                dosage TEXT NOT NULL,
                time TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
init_db()

# --- Medication Tracker Routes ---
@app.route("/medications", methods=["GET", "POST"])
def medications():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        dosage = request.form.get("dosage", "").strip()
        time = request.form.get("time", "").strip()
        if name and dosage and time:
            cursor.execute(
                "INSERT INTO medications (name, dosage, time) VALUES (?, ?, ?)",
                (name, dosage, time)
            )
            conn.commit()

    # Fetch only date (not time)
    cursor.execute("""
        SELECT id, name, dosage, time, DATE(created_at) as date_only
        FROM medications ORDER BY created_at DESC
    """)
    meds = cursor.fetchall()
    conn.close()

    return render_template("medications.html", meds=meds)

@app.route("/delete_med/<int:mid>")
def delete_med(mid):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM medications WHERE id=?", (mid,))
    conn.commit()
    conn.close()
    return redirect(url_for("medications"))

MODEL_DIR = Path("model")

# --- Load per-disease models ---
def load_model_for(slug):
    mapping = {
        "diabetes": "diabetes",
        "flu": "flu",
        "pneumonia": "pneumonia",
        "heart": "heart_disease",
        "kidney": "ckd"
    }
    key = mapping.get(slug)
    if not key:
        raise ValueError(f"No model for slug {slug}")

    model = joblib.load(MODEL_DIR / f"{key}_model.pkl")
    imputer = joblib.load(MODEL_DIR / f"{key}_imputer.pkl")
    with open(MODEL_DIR / f"{key}_features.json") as f:
        features = json.load(f)

    return model, imputer, features

# --- Helper functions ---
def risk_msg(prob):
    if prob < 0.30:
        return ("✅ Low risk – No need to worry", "ok", prob)
    elif prob < 0.60:
        return ("⚠️ Moderate risk – Keep monitoring", "moderate", prob)
    else:
        return ("🚨 High risk – Please consult a doctor", "high", prob)

def plot_probs(labels, probs):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(labels, probs, color="#3a62ff")
    ax.set_xlabel("Probability")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded

# --- PWA Routes ---
@app.route('/manifest.json')
def serve_manifest():
    return app.send_static_file('manifest.json')

@app.route('/sw.js')
def serve_sw():
    response = app.send_static_file('sw.js')
    response.headers['Content-Type'] = 'application/javascript'
    return response

# --- ROUTES ---
@app.route("/")
def home():
    session["history"] = []
    return render_template("index.html")

@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    if request.method == "POST":
        user_input = request.json.get("message", "").strip()
        reply = query_gemini(user_input)
        return jsonify({"reply": reply})
    return render_template("chatbot.html")

@app.route("/mental_health")
def mental_health():
    return render_template("mental_health.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/hospitals")
def hospitals():
    return render_template("hospitals.html")

@app.route("/diseases")
def diseases():
    cards = [
        {"slug":"diabetes","title":"Diabetes","img":"images/dia.jpg","desc":"Glucose, blood pressure & BMI based screening."},
        {"slug":"flu","title":"Flu (Influenza)","img":"images/flu.jpeg","desc":"Fever, cough, chills, headache."},
        {"slug":"pneumonia","title":"Pneumonia","img":"images/pnue.jpeg","desc":"Chest pain, fever, shortness of breath."},
        {"slug":"heart","title":"Heart Disease","img":"images/heartattack.jpg","desc":"Cholesterol, BP, max heart rate, chest pain type."},
        {"slug":"kidney","title":"Kidney Issues","img":"images/kidney2.png","desc":"BP, swelling, fatigue and urinary symptoms."},
    ]
    return render_template("diseases.html", cards=cards)

@app.route("/encyclopedia")
def encyclopedia():
    return render_template("encyclopedia.html")

@app.route("/firstaid")
def firstaid():
    return render_template("firstaid.html")

@app.route("/dashboard")
def dashboard():
    history = session.get("history", [])
    plot_url = None
    if history:
        labels = [f"{i+1}. {h['disease']}" for i, h in enumerate(history)]
        probs = [h["prob"] for h in history]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(1, len(probs)+1), probs, marker="o", color="#3a62ff")
        ax.set_title("Prediction History")
        ax.set_xlabel("Prediction #")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        plt.xticks(range(1, len(probs)+1), labels, rotation=45, ha="right")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plot_url = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
    return render_template("dashboard.html", history=history, plot_url=plot_url)

@app.route("/bmi", methods=["GET","POST"])
def bmi():
    result = None
    if request.method == "POST":
        weight = float(request.form.get("weight", 0))
        height = float(request.form.get("height", 0)) / 100
        if height > 0:
            bmi_val = weight / (height**2)
            if bmi_val < 18.5: category = "Underweight"
            elif bmi_val < 25: category = "Normal"
            elif bmi_val < 30: category = "Overweight"
            else: category = "Obese"
            result = {"bmi": round(bmi_val,1), "category": category}
    return render_template("bmi.html", result=result)

@app.route("/bodymap")
def bodymap():
    return render_template("bodymap.html")

@app.route("/history")
def history():
    data = session.get("history", [])
    return render_template("history.html", history=data)

# --- Prediction Route ---
@app.route("/predict/<slug>", methods=["GET","POST"])
def predict_slug(slug):
    slug = slug.lower()
    if request.method == "GET":
        return render_template(f"predict_{slug}.html")

    user_features = {}
    if slug == "diabetes":
        user_features = {
            "age": request.form.get("age", 0),
            "glucose": request.form.get("glucose", 0),
            "blood_pressure": request.form.get("bp", 0),
            "bmi": request.form.get("bmi", 0),
            "pregnancies": request.form.get("pregnancies", 0),
            "fatigue": 1 if request.form.get("fatigue") == "on" else 0,
            "weight_loss": 1 if request.form.get("weight_loss") == "on" else 0,
            "excessive_hunger": 1 if request.form.get("excessive_hunger") == "on" else 0,
            "frequent_urination": 1 if request.form.get("polyuria") == "on" else 0,
        }
    elif slug == "flu":
        user_features = {
            "high_fever": 1 if request.form.get("high_fever") == "on" else 0,
            "body_ache": 1 if request.form.get("body_ache") == "on" else 0,
            "chills": 1 if request.form.get("chills") == "on" else 0,
            "cough": 1 if request.form.get("cough") == "on" else 0,
            "headache": 1 if request.form.get("headache") == "on" else 0,
            "fatigue": 1 if request.form.get("fatigue") == "on" else 0,
        }
    elif slug == "pneumonia":
        user_features = {
            "high_fever": 1 if request.form.get("high_fever") == "on" else 0,
            "productive_cough": 1 if request.form.get("productive_cough") == "on" else 0,
            "shortness_of_breath": 1 if request.form.get("shortness_of_breath") == "on" else 0,
            "chest_pain": 1 if request.form.get("chest_pain") == "on" else 0,
        }
    elif slug == "heart":
        user_features = {
            "age": request.form.get("age", 0),
            "blood_pressure": request.form.get("bp", 0),
            "cholesterol": request.form.get("cholesterol", 0),
            "max_heart_rate": request.form.get("max_heart_rate", 0),
            "chest_discomfort": 1 if request.form.get("chest_discomfort") == "on" else 0,
            "shortness_of_breath": 1 if request.form.get("shortness_of_breath") == "on" else 0,
            "dizziness": 1 if request.form.get("dizziness") == "on" else 0,
        }
    elif slug == "kidney":
        user_features = {
            "blood_pressure": request.form.get("bp", 0),
            "fatigue": 1 if request.form.get("fatigue") == "on" else 0,
            "lower_abdominal_pain": 1 if request.form.get("lower_abdominal_pain") == "on" else 0,
            "burning_urination": 1 if request.form.get("burning_urination") == "on" else 0,
            "leg_swelling": 1 if request.form.get("leg_swelling") == "on" else 0,
        }

    model, imputer, features = load_model_for(slug)

    row = {col: 0 for col in features}
    for k, v in user_features.items():
        try:
            row[k] = float(v)
        except:
            row[k] = 0

    df = pd.DataFrame([row], columns=features)

    df_imp = imputer.transform(df)
    proba = model.predict_proba(df_imp)[0]

    disease_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
    message, tag, _ = risk_msg(disease_prob)

    history = session.get("history", [])
    history.append({"disease": slug.title(), "prob": disease_prob})
    session["history"] = history

    labels = [slug.title()]
    probs = [disease_prob]
    plot_url = plot_probs(labels, probs)

    return render_template("result.html",
                           tag=tag, message=message,
                           prob_percent=f"{disease_prob*100:.1f}%",
                           top3=[(slug.title(), disease_prob)],
                           plot_url=plot_url)

if __name__ == "__main__":
    app.run(debug=True)
