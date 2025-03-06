# 📄 AI Based Resume Screening App

## 🔍 Overview
The **Resume Screening App** is a machine-learning-powered tool that classifies resumes into different job categories based on their content. The application utilizes **Natural Language Processing (NLP)** techniques and a trained **Logistic Regression** and **Random Forest model** to make predictions.

## Demo
Try it out at - [Link](https://ai-resume-screening-app.streamlit.app/)

## 🚀 Features
- 📂 Upload a resume (Text/PDF/DOCX)
- 🔍 Extract key details (Name, Contact, Education, Skills, Education Detalis)
- 🎯 Predict the job category
- 📊 Interactive **Streamlit** UI for real-time predictions

## 🏗️ Tech Stack
- **Python** 🐍
- **Streamlit** (Web UI)
- **Scikit-learn** (ML models)
- **NLTK & SpaCy** (Text Processing)
- **Pandas & NumPy** (Data Handling)
- **Pickle** (Model Storage)

## 📂 Directory Structure
```
└── arpitkadam-resume-screening-app/
    ├── Resume_Categorization.ipynb   # Notebook for training and testing models
    ├── main.py                        # Streamlit web app
    ├── requirements.txt               # Dependencies
    ├── data/
    │   └── clean_resume_data.csv       # Preprocessed resume dataset
    ├── images/                         # UI and result snapshots (optional)
    ├── models/
    │   ├── best_LogisticRegression_model.pkl  # Logistic Regression Model
    │   ├── best_RandomForest_model.pkl        # Random Forest Model
    │   └── vectorizer.pkl                      # TF-IDF Vectorizer
    └── .streamlit/
        └── config.toml                         # Streamlit configurations
```

## ⚡ Installation & Usage
### 🔹 1. Clone the Repository
```sh
git clone https://github.com/ArpitKadam/Resume-Screening-App.git
cd Resume-Screening-App
```
### 🔹 2. Create a Virtual Environment (Optional but Recommended)
```sh
python -m venv venv
source venv/bin/activate  # For Mac/Linux
venv\Scripts\activate     # For Windows
```
### 🔹 3. Install Dependencies
```sh
pip install -r requirements.txt
```
### 🔹 4. Run the Application
```sh
streamlit run main.py
```

## 🖼️ Screenshots

### 📊 Countplot of Job Categories
![Countplot of Job Categories](https://github.com/ArpitKadam/Resume-Screening-App/blob/main/images/Countplot%20of%20Job%20Categories.png)
### 📊 Pieplot of Job Categories
![Piechart of Job Categories](https://github.com/ArpitKadam/Resume-Screening-App/blob/main/images/Pie%20Chart%20of%20Job%20Categories.png)

## 🏆 Models Used
- **Logistic Regression** ✅
- **Random Forest Classifier** 🌳
- **TF-IDF Vectorizer** for feature extraction ✨

## ✨ Features Implemented
✔️ Resume Text Processing (Name, Email, Contact, Skills, Education, Education Details.)
✔️ Job Category Prediction
✔️ Interactive Streamlit UI
✔️ Pretrained ML Models for Accuracy

## 🤝 Contributing
Pull requests are welcome! Feel free to open an issue for discussions.

## 📜 License
This project is open-source under the [**MIT License**](https://github.com/ArpitKadam/Resume-Screening-App/blob/main/LICENSE).

## 🔗 Connect with Me
[![Personal Website](https://img.shields.io/badge/Personal-4CAF50?style=for-the-badge&logo=googlechrome&logoColor=white)](https://arpit-kadam.netlify.app/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-arpitkadam-blue?logo=linkedin)](https://www.linkedin.com/in/arpitkadam/)
[![GitHub](https://img.shields.io/badge/GitHub-arpitkadam-black?logo=github)](https://github.com/arpitkadam)

🚀 **Happy Coding!** 🎯

