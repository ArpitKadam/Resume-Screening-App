import streamlit as st
import pandas as pd
import joblib
import re
import time
import PyPDF2
from io import StringIO

# Load Model & Vectorizer
model = joblib.load("models/best_LogisticRegression_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Function to Clean Resume Text
def clean_resume(txt):
    clean_text = re.sub('http\S+\s*', ' ', txt)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]',r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text.strip()

# Function to Predict Category
def predict_category(resume_text):
    resume_text = clean_resume(resume_text)
    resume_vectorized = vectorizer.transform([resume_text])
    predicted_category = model.predict(resume_vectorized)[0]
    return predicted_category

# Function to Extract Text from Uploaded PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# 🎨 Streamlit Page Configuration
st.set_page_config(page_title="AI Resume Screening", page_icon="📄", layout="wide")

# 🌟 Stylish Header
st.markdown("""
    <div style="text-align: center;">
        <h1 style="color: #4A90E2;">📄 AI Resume Screening App</h1>
        <p style="font-size: 18px;">Analyze your resume and predict the job category using AI.</p>
    </div>
""", unsafe_allow_html=True)

# 📂 File Upload & Text Area
uploaded_file = st.file_uploader("Upload your resume (PDF or TXT)", type=["pdf", "txt"])
resume_text = ""

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    else:
        resume_text = StringIO(uploaded_file.getvalue().decode("utf-8")).read()

resume_text = st.text_area("Or paste your resume text below:", resume_text, height=600)

# 🚀 Prediction Button with Progress Animation
if st.button("Predict Category"):
    if not resume_text.strip():
        st.warning("⚠️ Please provide resume text or upload a file.")
    else:
        with st.spinner("🔍 Analyzing Resume..."):
            time.sleep(2)  # Simulating Processing Time
            predicted_category = predict_category(resume_text)

        # 🎯 Display Prediction with Styled Badge
        st.markdown(f"""
            <div style="text-align: center;">
                <h3 style="color: #2ECC71;">✅ Predicted Job Category:</h3>
                <p style="background-color: #202121; padding: 10px; font-size: 22px; border-radius: 8px;">
                    {predicted_category}
                </p>
            </div>
        """, unsafe_allow_html=True)

# 🔗 Social Media Links
st.markdown("""
    <hr>
    <div style="text-align: center;">
        <p>🔹 Built with <b>Streamlit & Machine Learning</b> | 🎯 AI-Powered Resume Categorization 🔹</p>
        <br>
        <a href="https://arpit-kadam.netlify.app/" target="_blank">
            <img src="https://img.shields.io/badge/Personal-4CAF50?style=for-the-badge&logo=googlechrome&logoColor=white">
        </a>
        <a href="mailto:arpitkadam922@gmail.com">
            <img src="https://img.shields.io/badge/gmail-D14836?&style=for-the-badge&logo=gmail&logoColor=white">
        </a>
        <a href="https://www.linkedin.com/in/arpitkadam/" target="_blank">
            <img src="https://img.shields.io/badge/LinkedIn-0077B5?&style=for-the-badge&logo=linkedin&logoColor=white">
        </a>
        <a href="https://dagshub.com/ArpitKadam" target="_blank">
            <img src="https://img.shields.io/badge/DAGsHub-231F20?style=for-the-badge&logo=dagshub&logoColor=white">
        </a>
        <a href="https://twitter.com/arpitkadam5" target="_blank">
            <img src="https://img.shields.io/badge/Twitter-1DA1F2?&style=for-the-badge&logo=twitter&logoColor=white">
        </a>
        <a href="https://dev.to/arpitkadam" target="_blank">
            <img src="https://img.shields.io/badge/Dev.to-0A0A0A?&style=for-the-badge&logo=dev.to&logoColor=white">
        </a>
        <a href="https://www.kaggle.com/arpitkadam" target="_blank">
            <img src="https://img.shields.io/badge/Kaggle-20BEFF?&style=for-the-badge&logo=kaggle&logoColor=white">
        </a>
        <a href="https://www.instagram.com/arpit__kadam/" target="_blank">
            <img src="https://img.shields.io/badge/Instagram-E1306C?&style=for-the-badge&logo=instagram&logoColor=white">
        </a>
        <a href="https://github.com/arpitkadam" target="_blank">
            <img src="https://img.shields.io/badge/GitHub-181717?&style=for-the-badge&logo=github&logoColor=white">
        </a>
        <a href="https://buymeacoffee.com/arpitkadam" target="_blank">
            <img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-FB7A1E?style=for-the-badge&logo=buymeacoffee&logoColor=white">
        </a>
    </div>
""", unsafe_allow_html=True)
