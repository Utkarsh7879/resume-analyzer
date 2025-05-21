# resume-analyzer
# flask code 
import os
import re
import uuid
import pickle
import spacy
import docx
import pdfplumber
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_from_directory
from werkzeug.utils import secure_filename
from wordcloud import WordCloud
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Load NLP and ML model
nlp = spacy.load("en_core_web_sm")
loaded_data = pickle.load(open("resume_classifier.pkl", "rb"))

if isinstance(loaded_data, tuple):
    vectorizer, model = loaded_data
else:
    model = loaded_data
    vectorizer = None

# App configuration
app = Flask(__name__)
app.secret_key = 'supersecretkey123'
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# In-memory user database
users = {}

# -------------------- Text Extraction --------------------
def extract_text(file_path):
    ext = file_path.split('.')[-1].lower()
    text = ''
    try:
        if ext == 'pdf':
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ''
        elif ext == 'docx':
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + '\n'
    except Exception as e:
        print(f"Error extracting text: {e}")
    print(f"Extracted resume text length: {len(text)}")
    print(f"Sample: {text[:500]}")
    return text.strip()

# -------------------- Resume Analysis Logic --------------------
def extract_personal_details(text):
    doc = nlp(text)
    name, email, phone = "Unknown", "Not Found", "Not Found"

    email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    phone_match = re.search(r'(\+?\d{1,3}[-\s]?)?\(?\d{2,4}\)?[-\s]?\d{3,4}[-\s]?\d{3,4}', text)

    if email_match:
        email = email_match.group(0)
    if phone_match:
        phone = phone_match.group(0)

    for ent in doc.ents:
        if ent.label_ == "PERSON" and ent.text.lower() not in {"resume", "coursework", "skills"}:
            name = ent.text
            break

    if name == "Unknown" and email != "Not Found":
        name = email.split('@')[0].replace('.', ' ').replace('_', ' ').title()

    return {"Name": name, "Email": email, "Phone": phone}

def predict_category(text):
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    if vectorizer:
        vector = vectorizer.transform([cleaned_text])
        return model.predict(vector)[0]
    return "Unknown"

def estimate_experience(text):
    matches = re.findall(r'(\d+)\+?\s+years?', text.lower())
    return max([int(m) for m in matches]) if matches else 0

def generate_wordcloud(skills, session_id):
    img_path = f"static/wordcloud_{session_id}.jpg"
    if not skills:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, 'No skills to show', color='white', ha='center', va='center', fontsize=20)
        plt.gca().set_facecolor('black')
        plt.axis('off')
        plt.savefig(img_path, bbox_inches='tight')
        plt.close()
        return img_path

    wc = WordCloud(width=800, height=400, background_color="black").generate(" ".join(skills))
    wc.to_file(img_path)
    return img_path

def generate_pdf_report(data, session_id):
    filepath = f"static/resume_report_{session_id}.pdf"
    c = canvas.Canvas(filepath, pagesize=letter)
    t = c.beginText(50, 750)
    t.setFont("Helvetica", 12)
    for k, v in data.items():
        if isinstance(v, list):
            v = ", ".join(v)
        t.textLine(f"{k}: {v}")
    c.drawText(t)
    c.save()
    return filepath

def calculate_ats_score(text, present_skills, recommended_skills):
    if not recommended_skills:
        return 0
    match_score = len(present_skills) / max(len(recommended_skills), 1)
    length_score = min(1.0, len(text.split()) / 500)
    return int((0.7 * match_score + 0.3 * length_score) * 100)

def calculate_selection_chance(ats, exp):
    return min(100, int(ats * 0.6 + min(exp * 5, 30)))

def analyze_resume(text, session_id):
    details = extract_personal_details(text)
    category = predict_category(text)
    print(f"[DEBUG] Initial Predicted Category: {category}")  # Debug log

    skill_map = {
        "Data Science": ["Python", "Machine Learning", "data analyst", "Power BI"],
        "Software Engineering": ["Java", "C++", "React", "Spring", "Git"],
        "Web Development": ["HTML", "CSS", "JavaScript", "Node.js", "MongoDB"],
        "Android Developer": ["Kotlin", "Android Studio", "Firebase"],
        "Cybersecurity": ["Nmap", "Burp Suite", "Firewall", "Encryption", "Wireshark"],
        "DevOps": ["Docker", "Kubernetes", "CI/CD", "AWS"],
        "UI/UX": ["Figma", "Adobe XD", "User Research"],
    }

    recommended_skills = skill_map.get(category, [])

    # Fallback: Try to guess category based on skill presence
    if not recommended_skills:
        for cat, skills in skill_map.items():
            if any(skill.lower() in text.lower() for skill in skills):
                print(f"[DEBUG] Fallback Category Matched: {cat}")
                category = cat
                recommended_skills = skills
                break

    # Optional: default fallback category
    if not recommended_skills:
        category = "General"
        recommended_skills = ["Communication", "Problem Solving", "Teamwork"]

    present_skills = [skill for skill in recommended_skills if skill.lower() in text.lower()]
    exp_years = estimate_experience(text)
    ats_score = calculate_ats_score(text, present_skills, recommended_skills)
    selection_chance = calculate_selection_chance(ats_score, exp_years)

    wordcloud_path = generate_wordcloud(recommended_skills, session_id)
    pdf_path = generate_pdf_report({
        **details,
        "Category": category,
        "Experience": f"{exp_years} years",
        "Skills": present_skills,
        "ATS Score": ats_score,
        "Recommended Skills": recommended_skills,
        "Selection Chance": selection_chance
    }, session_id)

    return {
        **details,
        "Category": category,
        "Experience": f"{exp_years} years",
        "Skills": present_skills,
        "ATS Score": ats_score,
        "Recommended Skills": recommended_skills,
        "Selection Chance": selection_chance,
        "WordCloud": wordcloud_path,
        "PDF Report": pdf_path
    }


# -------------------- Routes --------------------
@app.route('/')
def home():
    if "user" not in session:
        return redirect(url_for("auth_page"))
    return render_template("index.html", username=session["user"])

@app.route('/auth')
def auth_page():
    return render_template("auth.html")

@app.route("/signup", methods=["POST"])
def signup():
    u, p = request.form["signup_username"], request.form["signup_password"]
    if u in users:
        return render_template("auth.html", message="User already exists.")
    users[u] = p
    session["user"] = u
    return redirect(url_for("home"))

@app.route("/login", methods=["POST"])
def login():
    u, p = request.form["login_username"], request.form["login_password"]
    if users.get(u) == p:
        session["user"] = u
        return redirect(url_for("home"))
    return render_template("auth.html", message="Invalid credentials.")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("auth_page"))

@app.route("/upload", methods=["POST"])
def upload():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    file = request.files.get("resume")
    if not file or file.filename == '':
        return jsonify({"error": "No file uploaded"}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        text = extract_text(filepath)
        if not text.strip():
            return jsonify({"error": "Unable to extract text"}), 400

        session_id = str(uuid.uuid4())
        result = analyze_resume(text, session_id)
        session["report_path"] = result["PDF Report"]
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Error analyzing resume: {str(e)}"}), 500

@app.route("/download_report")
def download_report():
    report_path = session.get("report_path")
    if not report_path or not os.path.exists(report_path):
        return "Report not found", 404
    return send_from_directory("static", os.path.basename(report_path), as_attachment=True)

# -------------------- Run App --------------------
if __name__ == "__main__":
    app.run(debug=True)
