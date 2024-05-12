import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
import pdfplumber
import spacy

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(os.getcwd(), "uploads")

# Load CSV data
csv_file_path = "C:/Users/USER/PycharmProjects/Resume_parser.py/static/ResumeDataSet.csv"
df = pd.read_csv(csv_file_path)

# Train a machine learning model (you may need to adjust this based on your data and model)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Resume'])
y = df['Category']

model = LogisticRegression()
model.fit(X, y)

# Set the secret key
app.secret_key = os.urandom(24)

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the desired job position from the form
        job_position = request.form.get("job_position")

        # Redirect to the upload page with the job position as a query parameter
        return redirect(url_for("upload", job_position=job_position))

    return render_template("index.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    required_skills = request.args.get("required_skills")
    required_experiences = request.args.get("required_experiences")
    required_skills = required_skills.split(",") if required_skills else []
    required_experiences = required_experiences.split(",") if required_experiences else []

    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            # Save the uploaded file to a temporary location
            uploaded_file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(uploaded_file_path)

            resume_text = extract_text_from_pdf(uploaded_file_path)
            doc = nlp(resume_text)

            extracted_skills = []
            extracted_experiences = []
            for ent in doc.ents:
                if "skill" in ent.text.lower() and any(skill.lower() in ent.text.lower() for skill in required_skills):
                    extracted_skills.append(ent.text)
                if "experience" in ent.text.lower() and any(
                        experience.lower() in ent.text.lower() for experience in required_experiences):
                    extracted_experiences.append(ent.text)

            # If "experience" not found, set count to 0
            if len(extracted_experiences) == 0:
                extracted_experiences_count = 0
            else:
                extracted_experiences_count = len(extracted_experiences)

            # Check if all required skills and experiences are found
            all_skills_found = all(skill.lower() in [s.lower() for s in extracted_skills] for skill in required_skills)
            all_experiences_found = all(
                experience.lower() in [e.lower() for e in extracted_experiences] for experience in required_experiences)

            match = len(extracted_skills) > 0 and len(extracted_experiences) > 0
            return render_template("result.html",
                                   job_position=request.args.get("job_position"),
                                   predicted_job=model.predict(vectorizer.transform([resume_text]))[0],
                                   extracted_skills=extracted_skills,
                                   extracted_experiences=extracted_experiences,
                                   extracted_experiences_count=extracted_experiences_count,
                                   all_skills_found=all_skills_found,
                                   all_experiences_found=all_experiences_found,
                                   match=match)

    return render_template("front_page.html")


# Clear the uploaded files in the "uploads" folder
@app.route("/clear", methods=["POST"])
def clear_uploads():
    folder = "uploads"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting file: {e}")
    return redirect(url_for("index"))


def extract_text_from_pdf(pdf_file):
    """
    Function to extract text from a PDF file using pdfplumber.
    """
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

if __name__ == "__main__":
    app.run(debug=True)
