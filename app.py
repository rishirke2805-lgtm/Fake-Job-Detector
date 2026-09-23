from flask import Flask, render_template, request, jsonify
import joblib
import sqlite3
import json
import pytesseract
from PIL import Image

from backend.rule_engine import check_red_flags

# NOTE: Tell Python exactly where Tesseract is installed on your Windows machine
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__, template_folder='frontend')

model = joblib.load("model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def init_db():
    with sqlite3.connect("database.db") as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS job_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_text TEXT,
                prediction TEXT,
                confidence TEXT,
                red_flags TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

init_db()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    job_text = ""

    # 1. Check if an image file was uploaded
    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            # Open the image and extract text using OCR
            img = Image.open(file.stream)
            job_text = pytesseract.image_to_string(img)

    # 2. If no image was uploaded, check for typed text
    if not job_text:
        job_text = request.form.get("text", "")
    
    if not job_text.strip():
        return jsonify({"error": "Please enter text or upload a valid screenshot."})

    # ML Prediction
    text_tfidf = vectorizer.transform([job_text])
    prediction = model.predict(text_tfidf)[0]
    
    probabilities = model.predict_proba(text_tfidf)[0]
    confidence = round(probabilities[1] * 100, 2) if prediction == 1 else round(probabilities[0] * 100, 2)
    result_label = "Fraudulent" if prediction == 1 else "Genuine"

    # Rule Engine Assessment
    red_flags = check_red_flags(job_text)

    # Save to SQLite
    with sqlite3.connect("database.db") as conn:
        cursor = conn.cursor()
        flags_json = json.dumps(red_flags)
        cursor.execute('''
            INSERT INTO job_checks (job_text, prediction, confidence, red_flags)
            VALUES (?, ?, ?, ?)
        ''', (job_text, result_label, f"{confidence}%", flags_json))
        conn.commit()

    return jsonify({
        "prediction": result_label,
        "confidence": f"{confidence}%",
        "red_flags": red_flags,
        "extracted_text": job_text # Send the OCR text back so the user can see what the AI read
    })

@app.route('/logs', methods=['GET'])
def get_logs():
    with sqlite3.connect("database.db") as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM job_checks ORDER BY timestamp DESC LIMIT 20")
        rows = cursor.fetchall()
        
        logs = []
        for row in rows:
            is_fraud = row["prediction"] == "Fraudulent"
            logs.append({
                "snippet": row["job_text"][:45].replace('\n', ' ') + "...",
                "score": row["confidence"].replace('%', ''),
                "verdict": "Suspicious" if is_fraud else "Looks Genuine",
                "tone": "danger" if is_fraud else "safe",
                "time": row["timestamp"] 
            })
            
    return jsonify(logs)

if __name__ == "__main__":
    app.run(debug=True)