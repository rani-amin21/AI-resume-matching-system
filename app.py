import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
import spacy
import PyPDF2
import pickle

from sklearn.metrics.pairwise import cosine_similarity

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(page_title="AI Recruitment & Job Matching Platform", layout="wide")

# ==================================================
# CUSTOM CSS
# ==================================================
st.markdown("""
<style>
body {
    background-color: #0b1120;
}
.main {
    background: linear-gradient(135deg,#0f172a,#0b1120);
    color: white;
}
.block-container {
    padding-top: 40px;
}
.hero {
    text-align: center;
}
.hero h1 {
    font-size: 60px;
    font-weight: 700;
}
.hero h2 {
    font-size: 28px;
}
button[data-baseweb="tab"] {
    font-size: 20px !important;
    font-weight: 600 !important;
}
.card {
    background: #111827;
    padding: 25px;
    border-radius: 14px;
    font-size: 18px;
}
.card h4 {
    font-size: 22px;
}
.stButton > button {
    font-size: 18px !important;
    padding: 12px 24px !important;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# TITLE
# ==================================================
st.markdown("""
<div class="hero">
    <h1>AI Recruitment & Job Matching Platform</h1>
    <h2>Smart Matching for Candidates. Intelligent Screening for Recruiters.</h2>
</div>
""", unsafe_allow_html=True)

# ==================================================
# LOAD NLP
# ==================================================
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

# ==================================================
# LOAD PICKLE FILES
# ==================================================
@st.cache_resource
def load_model_files():
    with open("tfidf.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("job_role_tfidf.pkl", "rb") as f:
        job_vectors = pickle.load(f)
    with open("job_role_df.pkl", "rb") as f:
        job_roles = pickle.load(f)
    return vectorizer, job_vectors, job_roles

vectorizer, job_vectors, job_roles = load_model_files()

# ==================================================
# TEXT CLEANING
# ==================================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

# ==================================================
# PDF EXTRACT
# ==================================================
def extract_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# ==================================================
# MAIN TABS
# ==================================================
tab1, tab2, tab3 = st.tabs(["How It Works", "Candidate", "Recruiter"])

# ==================================================
# HOW IT WORKS (Compact Layout)
# ==================================================
with tab1:

    st.markdown("## How It Works")

    st.markdown("### 👩‍💼 For Candidate")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card"><h4>1️⃣ Upload Resume</h4><p>Upload PDF or paste resume text.</p></div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><h4>2️⃣ AI Skill Analysis</h4><p>TF-IDF converts resume into vectors.</p></div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card"><h4>3️⃣ Get Job Matches</h4><p>Cosine similarity ranks job roles.</p></div>', unsafe_allow_html=True)

    st.markdown("### 🏢 For Recruiter")
    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown('<div class="card"><h4>1️⃣ Paste Job Description</h4><p>Enter job requirements.</p></div>', unsafe_allow_html=True)

    with col5:
        st.markdown('<div class="card"><h4>2️⃣ Upload Resume</h4><p>Upload or paste candidate resume.</p></div>', unsafe_allow_html=True)

    with col6:
        st.markdown('<div class="card"><h4>3️⃣ Get Match Score</h4><p>System recommends hire decision.</p></div>', unsafe_allow_html=True)

# ==================================================
# CANDIDATE TAB
# ==================================================
with tab2:

    st.markdown("## 📤 Upload Your Resume")

    resume_text = ""

    upload_tab1, upload_tab2 = st.tabs(["Upload PDF", "Paste Text"])

    with upload_tab1:
        uploaded_file = st.file_uploader("Upload PDF Resume", type=["pdf"])
        if uploaded_file:
            resume_text = extract_pdf(uploaded_file)

    with upload_tab2:
        resume_text_input = st.text_area("Paste Resume Text Here")
        if resume_text_input:
            resume_text = resume_text_input

    if st.button("🔍 Find Matching Jobs"):

        if resume_text.strip() == "":
            st.warning("Please upload or paste resume first.")
            st.stop()

        cleaned_resume = clean_text(resume_text)
        resume_vector = vectorizer.transform([cleaned_resume])
        similarity = cosine_similarity(resume_vector, job_vectors)[0]

        top_indices = similarity.argsort()[::-1][:10]

        results = pd.DataFrame({
            "Job Role": job_roles.iloc[top_indices]["Title"].values,
            "Match Score": similarity[top_indices].round(3)
        })

        st.markdown("### 💼 Recommended Job Roles")
        st.dataframe(results, use_container_width=True)

        # ================= Rapid API =================
        st.markdown("### 🌍 Real-Time Job Openings")

        RAPID_API_KEY = st.secrets["RAPID_API_KEY"]
        RAPID_API_HOST = st.secrets["RAPID_API_HOST"]

        predicted_role = results["Job Role"].iloc[0]

        url = "https://jsearch.p.rapidapi.com/search"

        querystring = {
            "query": predicted_role + " in India",
            "page": "1",
            "num_pages": "1"
        }

        headers = {
            "X-RapidAPI-Key": RAPID_API_KEY,
            "X-RapidAPI-Host": RAPID_API_HOST
        }

        response = requests.get(url, headers=headers, params=querystring)

        if response.status_code == 200:
            data = response.json()
            jobs = data.get("data", [])

            if jobs:
                job_list = []
                for job in jobs[:10]:
                    job_list.append({
                        "Title": job.get("job_title"),
                        "Company": job.get("employer_name"),
                        "Location": job.get("job_city"),
                        "Apply Link": job.get("job_apply_link")
                    })

                st.dataframe(pd.DataFrame(job_list), use_container_width=True)
            else:
                st.info("No live jobs found.")
        else:
            st.error("API Error. Check your RapidAPI key.")

# ==================================================
# RECRUITER TAB
# ==================================================
with tab3:

    st.markdown("## 🏢 Recruiter Candidate Evaluation")

    job_description = st.text_area("📄 Paste Job Description Here")

    st.markdown("### 📤 Candidate Resume")

    rec_tab1, rec_tab2 = st.tabs(["Upload PDF", "Paste Text"])

    recruiter_resume_text = ""

    with rec_tab1:
        uploaded_resume = st.file_uploader("Upload Candidate Resume PDF", type=["pdf"])
        if uploaded_resume:
            recruiter_resume_text = extract_pdf(uploaded_resume)

    with rec_tab2:
        resume_input = st.text_area("Paste Candidate Resume Here")
        if resume_input:
            recruiter_resume_text = resume_input

    if st.button("🔎 Evaluate Candidate"):

        if job_description.strip() == "" or recruiter_resume_text.strip() == "":
            st.warning("Please provide both Job Description and Resume.")
            st.stop()

        cleaned_job = clean_text(job_description)
        cleaned_resume = clean_text(recruiter_resume_text)

        job_vector = vectorizer.transform([cleaned_job])
        resume_vector = vectorizer.transform([cleaned_resume])

        similarity = cosine_similarity(resume_vector, job_vector)[0][0]
        score_10 = round(similarity * 10, 2)

        # Color coding
        if score_10 >= 7:
            color = "#22c55e"
        elif score_10 >= 4:
            color = "#facc15"
        else:
            color = "#ef4444"

        st.markdown("### 📊 Match Score")

        st.markdown(f"""
        <div style="background:#1f2937; padding:15px; border-radius:10px;">
            <div style="font-size:20px; margin-bottom:10px;">
                Score: <span style="color:{color}; font-weight:bold;">{score_10} / 10</span>
            </div>
            <div style="background:#374151; height:18px; border-radius:10px;">
                <div style="width:{score_10*10}%; 
                            background:{color}; 
                            height:18px; 
                            border-radius:10px;">
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if score_10 >= 7:
            st.success("✅ Strong Match – Recommended to Hire")
        elif score_10 >= 4:
            st.warning("⚠ Moderate Match – Consider for Interview")
        else:
            st.error("❌ Weak Match – Not Recommended")