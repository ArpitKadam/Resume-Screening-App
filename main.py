import streamlit as st
import pandas as pd
import joblib
import re
import time
import PyPDF2
from io import StringIO
import re

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

def extract_contact_number(text):
  contact_no = None
  pattern = r"(?:(?:\+?([1-9]|[0-9][0-9]|[0-9][0-9][0-9])\s*(?:[.-]\s*)?)?(?:\(\s*([2-9])))"
  match = re.search(pattern, text)
  if match:
    contact_no = match.group(0)
  return contact_no

def extract_email(text):
    """Extracts an email address from the resume text."""
    pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    match = re.search(pattern, text)
    return match.group(0) if match else None

def extract_experience(text):
    """Extracts the total years of experience mentioned in the resume."""
    pattern = r"(\d+)\s*(?:\+|\-)?\s*(?:years|yrs|year|yr)\s*(?:of\s*experience)?"
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    return max(map(int, matches)) if matches else None

def extract_projects(text):
    """Extracts project titles based on common formats like 'Project: XYZ' or 'Title: ABC'."""
    pattern = r"(?i)(?:(?:Project|Title|Research):?\s*)([A-Za-z0-9\s\-_,]+)"
    matches = re.findall(pattern, text)
    
    return [match.strip() for match in matches]


def extract_education_details(text):
  education_details = []
  pattern = r"(?i)(?:(?:Bachelor|Master|Doctorate|Diploma|PhD|B\.A\.|M\.A\.|B\.Sc\.|M\.Sc\.)\s*(?:of|in)?\s*([\w\s,]+))\s*(?:(?:from|at)\s*([\w\s,]+))?\s*(?:(?:\((?:\d{4}|\d{4}-\d{4})\))?)"
  matches = re.findall(pattern, text)

  for match in matches:
    degree = match[0].strip()
    institution = match[1].strip() if match[1] else None
    education_details.append({
        "degree": degree,
        "institution": institution,
    })
  return education_details

skills_list = [
    'Python', 'Data Analysis', 'Machine Learning', 'Communication', 'Project Management', 'Deep Learning', 'SQL', 'Tableau',
    'Java', 'C++', 'JavaScript', 'HTML', 'CSS', 'React', 'Angular', 'Node.js', 'MongoDB', 'Express.js', 'Git',
    'Research', 'Statistics', 'Quantitative Analysis', 'Qualitative Analysis', 'SPSS', 'R', 'Data Visualization', 'Matplotlib',
    'Seaborn', 'Plotly', 'Pandas', 'Numpy', 'Scikit-learn', 'TensorFlow', 'Keras', 'PyTorch', 'NLTK', 'Text Mining',
    'Natural Language Processing', 'Computer Vision', 'Image Processing', 'OCR', 'Speech Recognition', 'Recommendation Systems',
    'Collaborative Filtering', 'Content-Based Filtering', 'Reinforcement Learning', 'Neural Networks', 'Convolutional Neural Networks',
    'Recurrent Neural Networks', 'Generative Adversarial Networks', 'XGBoost', 'Random Forest', 'Decision Trees', 'Support Vector Machines',
    'Linear Regression', 'Logistic Regression', 'K-Means Clustering', 'Hierarchical Clustering', 'DBSCAN', 'Association Rule Learning',
    'Apache Hadoop', 'Apache Spark', 'MapReduce', 'Hive', 'HBase', 'Apache Kafka', 'Data Warehousing', 'ETL', 'Big Data Analytics',
    'Cloud Computing', 'Amazon Web Services (AWS)', 'Microsoft Azure', 'Google Cloud Platform (GCP)', 'Docker', 'Kubernetes', 'Linux',
    'Shell Scripting', 'Cybersecurity', 'Network Security', 'Penetration Testing', 'Firewalls', 'Encryption', 'Malware Analysis',
    'Digital Forensics', 'CI/CD', 'DevOps', 'Agile Methodology', 'Scrum', 'Kanban', 'Continuous Integration', 'Continuous Deployment',
    'Software Development', 'Web Development', 'Mobile Development', 'Backend Development', 'Frontend Development', 'Full-Stack Development',
    'UI/UX Design', 'Responsive Design', 'Wireframing', 'Prototyping', 'User Testing', 'Adobe Creative Suite', 'Photoshop', 'Illustrator',
    'InDesign', 'Figma', 'Sketch', 'Zeplin', 'InVision', 'Product Management', 'Market Research', 'Customer Development', 'Lean Startup',
    'Business Development', 'Sales', 'Marketing', 'Content Marketing', 'Social Media Marketing', 'Email Marketing', 'SEO', 'SEM', 'PPC',
    'Google Analytics', 'Facebook Ads', 'LinkedIn Ads', 'Lead Generation', 'Customer Relationship Management (CRM)', 'Salesforce',
    'HubSpot', 'Zendesk', 'Intercom', 'Customer Support', 'Technical Support', 'Troubleshooting', 'Ticketing Systems', 'ServiceNow',
    'ITIL', 'Quality Assurance', 'Manual Testing', 'Automated Testing', 'Selenium', 'JUnit', 'Load Testing', 'Performance Testing',
    'Regression Testing', 'Black Box Testing', 'White Box Testing', 'API Testing', 'Mobile Testing', 'Usability Testing', 'Accessibility Testing',
    'Cross-Browser Testing', 'Agile Testing', 'User Acceptance Testing', 'Software Documentation', 'Technical Writing', 'Copywriting',
    'Editing', 'Proofreading', 'Content Management Systems (CMS)', 'WordPress', 'Joomla', 'Drupal', 'Magento', 'Shopify', 'E-commerce',
    'Payment Gateways', 'Inventory Management', 'Supply Chain Management', 'Logistics', 'Procurement', 'ERP Systems', 'SAP', 'Oracle',
    'Microsoft Dynamics', 'Tableau', 'Power BI', 'QlikView', 'Looker', 'Data Warehousing', 'ETL', 'Data Engineering', 'Data Governance',
    'Data Quality', 'Master Data Management', 'Predictive Analytics', 'Prescriptive Analytics', 'Descriptive Analytics', 'Business Intelligence',
    'Dashboarding', 'Reporting', 'Data Mining', 'Web Scraping', 'API Integration', 'RESTful APIs', 'GraphQL', 'SOAP', 'Microservices',
    'Serverless Architecture', 'Lambda Functions', 'Event-Driven Architecture', 'Message Queues', 'GraphQL', 'Socket.io', 'WebSockets'
    'Ruby', 'Ruby on Rails', 'PHP', 'Symfony', 'Laravel', 'CakePHP', 'Zend Framework', 'ASP.NET', 'C#', 'VB.NET', 'ASP.NET MVC', 'Entity Framework',
    'Spring', 'Hibernate', 'Struts', 'Kotlin', 'Swift', 'Objective-C', 'iOS Development', 'Android Development', 'Flutter', 'React Native', 'Ionic',
    'Mobile UI/UX Design', 'Material Design', 'SwiftUI', 'RxJava', 'RxSwift', 'Django', 'Flask', 'FastAPI', 'Falcon', 'Tornado', 'WebSockets',
    'GraphQL', 'RESTful Web Services', 'SOAP', 'Microservices Architecture', 'Serverless Computing', 'AWS Lambda', 'Google Cloud Functions',
    'Azure Functions', 'Server Administration', 'System Administration', 'Network Administration', 'Database Administration', 'MySQL', 'PostgreSQL',
    'SQLite', 'Microsoft SQL Server', 'Oracle Database', 'NoSQL', 'MongoDB', 'Cassandra', 'Redis', 'Elasticsearch', 'Firebase', 'Google Analytics',
    'Google Tag Manager', 'Adobe Analytics', 'Marketing Automation', 'Customer Data Platforms', 'Segment', 'Salesforce Marketing Cloud', 'HubSpot CRM',
    'Zapier', 'IFTTT', 'Workflow Automation', 'Robotic Process Automation (RPA)', 'UI Automation', 'Natural Language Generation (NLG)',
    'Virtual Reality (VR)', 'Augmented Reality (AR)', 'Mixed Reality (MR)', 'Unity', 'Unreal Engine', '3D Modeling', 'Animation', 'Motion Graphics',
    'Game Design', 'Game Development', 'Level Design', 'Unity3D', 'Unreal Engine 4', 'Blender', 'Maya', 'Adobe After Effects', 'Adobe Premiere Pro',
    'Final Cut Pro', 'Video Editing', 'Audio Editing', 'Sound Design', 'Music Production', 'Digital Marketing', 'Content Strategy', 'Conversion Rate Optimization (CRO)',
    'A/B Testing', 'Customer Experience (CX)', 'User Experience (UX)', 'User Interface (UI)', 'Persona Development', 'User Journey Mapping', 'Information Architecture (IA)',
    'Wireframing', 'Prototyping', 'Usability Testing', 'Accessibility Compliance', 'Internationalization (I18n)', 'Localization (L10n)', 'Voice User Interface (VUI)',
    'Chatbots', 'Natural Language Understanding (NLU)', 'Speech Synthesis', 'Emotion Detection', 'Sentiment Analysis', 'Image Recognition', 'Object Detection',
    'Facial Recognition', 'Gesture Recognition', 'Document Recognition', 'Fraud Detection', 'Cyber Threat Intelligence', 'Security Information and Event Management (SIEM)',
    'Vulnerability Assessment', 'Incident Response', 'Forensic Analysis', 'Security Operations Center (SOC)', 'Identity and Access Management (IAM)', 'Single Sign-On (SSO)',
    'Multi-Factor Authentication (MFA)', 'Blockchain', 'Cryptocurrency', 'Decentralized Finance (DeFi)', 'Smart Contracts', 'Web3', 'Non-Fungible Tokens (NFTs)']

def extract_skills_from_resume(text, skills_list):
    skills = []

    for skill in skills_list:
        pattern = r"\b{}\b".format(re.escape(skill))
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            skills.append(skill)
    return skills

def extract_education_from_resume(text):
    education = []

    # List of education keywords to match against
    education_keywords = [
        'Computer Science', 'Information Technology', 'Software Engineering', 'Electrical Engineering', 'Mechanical Engineering', 'Civil Engineering',
        'Chemical Engineering', 'Biomedical Engineering', 'Aerospace Engineering', 'Nuclear Engineering', 'Industrial Engineering', 'Systems Engineering',
        'Environmental Engineering', 'Petroleum Engineering', 'Geological Engineering', 'Marine Engineering', 'Robotics Engineering', 'Biotechnology',
        'Biochemistry', 'Microbiology', 'Genetics', 'Molecular Biology', 'Bioinformatics', 'Neuroscience', 'Biophysics', 'Biostatistics', 'Pharmacology',
        'Physiology', 'Anatomy', 'Pathology', 'Immunology', 'Epidemiology', 'Public Health', 'Health Administration', 'Nursing', 'Medicine', 'Dentistry',
        'Pharmacy', 'Veterinary Medicine', 'Medical Technology', 'Radiography', 'Physical Therapy', 'Occupational Therapy', 'Speech Therapy', 'Nutrition',
        'Sports Science', 'Kinesiology', 'Exercise Physiology', 'Sports Medicine', 'Rehabilitation Science', 'Psychology', 'Counseling', 'Social Work',
        'Sociology', 'Anthropology', 'Criminal Justice', 'Political Science', 'International Relations', 'Economics', 'Finance', 'Accounting', 'Business Administration',
        'Management', 'Marketing', 'Entrepreneurship', 'Hospitality Management', 'Tourism Management', 'Supply Chain Management', 'Logistics Management',
        'Operations Management', 'Human Resource Management', 'Organizational Behavior', 'Project Management', 'Quality Management', 'Risk Management',
        'Strategic Management', 'Public Administration', 'Urban Planning', 'Architecture', 'Interior Design', 'Landscape Architecture', 'Fine Arts',
        'Visual Arts', 'Graphic Design', 'Fashion Design', 'Industrial Design', 'Product Design', 'Animation', 'Film Studies', 'Media Studies',
        'Communication Studies', 'Journalism', 'Broadcasting', 'Creative Writing', 'English Literature', 'Linguistics', 'Translation Studies',
        'Foreign Languages', 'Modern Languages', 'Classical Studies', 'History', 'Archaeology', 'Philosophy', 'Theology', 'Religious Studies',
        'Ethics', 'Education', 'Early Childhood Education', 'Elementary Education', 'Secondary Education', 'Special Education', 'Higher Education',
        'Adult Education', 'Distance Education', 'Online Education', 'Instructional Design', 'Curriculum Development'
        'Library Science', 'Information Science', 'Computer Engineering', 'Software Development', 'Cybersecurity', 'Information Security',
        'Network Engineering', 'Data Science', 'Data Analytics', 'Business Analytics', 'Operations Research', 'Decision Sciences',
        'Human-Computer Interaction', 'User Experience Design', 'User Interface Design', 'Digital Marketing', 'Content Strategy',
        'Brand Management', 'Public Relations', 'Corporate Communications', 'Media Production', 'Digital Media', 'Web Development',
        'Mobile App Development', 'Game Development', 'Virtual Reality', 'Augmented Reality', 'Blockchain Technology', 'Cryptocurrency',
        'Digital Forensics', 'Forensic Science', 'Criminalistics', 'Crime Scene Investigation', 'Emergency Management', 'Fire Science',
        'Environmental Science', 'Climate Science', 'Meteorology', 'Geography', 'Geomatics', 'Remote Sensing', 'Geoinformatics',
        'Cartography', 'GIS (Geographic Information Systems)', 'Environmental Management', 'Sustainability Studies', 'Renewable Energy',
        'Green Technology', 'Ecology', 'Conservation Biology', 'Wildlife Biology', 'Zoology']

    for keyword in education_keywords:
        pattern = r"(?i)\b{}\b".format(re.escape(keyword))
        match = re.search(pattern, text)
        if match:
            education.append(match.group())

    return education

def extract_name_from_resume(text):
    name = None

    # Use regex pattern to find a potential name
    pattern = r"(\b[A-Z][a-z]+\b)\s(\b[A-Z][a-z]+\b)"
    match = re.search(pattern, text)
    if match:
        name = match.group()

    return name

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
    
    name = extract_name_from_resume(resume_text)
    phone = extract_contact_number(resume_text)
    mail = extract_email(resume_text)
    extracted_education = extract_education_from_resume(resume_text)
    extracted_skills = extract_skills_from_resume(resume_text, skills_list)
    education_info = extract_education_details(resume_text)

    col1, col2 = st.columns(2)
    with col1:
            st.subheader("👤 Personal Details")
            st.write(f"**Name:** {name if name else 'Not found'}")
            st.write(f"**📞 Contact:** {phone if phone else 'Not found'}")

            st.subheader("📧 Email")
            if mail:
                st.write(f"- {mail}")
            else:
                st.write("Not found")

            st.subheader("🎓 Education")
            if extracted_education:
                for edu in extracted_education:
                    st.write(f"- {edu}")
            else:
                st.write("Not found")
            
            st.subheader("🎓 Education Details")
            if education_info:
                for edu in education_info:
                    st.write(f"- {edu}")
            else:
                st.write("Not found")

    with col2:
            st.subheader("🛠️ Skills")
            if extracted_skills:
                for skill in extracted_skills:
                    st.write(f"- {skill}")
            else:
                st.write("Not found")

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
