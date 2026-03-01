# 🤖 AI Recruitment & Job Matching Platform

An AI-powered Resume–Job Matching system that intelligently matches candidates to job roles using Natural Language Processing (NLP) and Machine Learning techniques.

🔗 **Live App:** https://ai-resume-matching-system.streamlit.app  

---

## 🚀 Features

### 👩‍💼 Candidate Module
- Upload PDF resume or paste text
- Resume preprocessing using spaCy
- TF-IDF vectorization
- Cosine similarity ranking
- Top 10 recommended job roles
- Real-time job listings via RapidAPI

### 🏢 Recruiter Module
- Paste job description
- Upload candidate resume
- Intelligent match scoring (0–10 scale)
- Visual match indicator
- Hire recommendation system

---

## 🧠 Technologies Used

- Python
- Streamlit
- spaCy (NLP)
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity
- Pandas & NumPy
- RapidAPI Integration

---

## 🏗 How It Works

1. Text cleaning & lemmatization using spaCy
2. TF-IDF converts text into numerical vectors
3. Cosine similarity measures resume-job relevance
4. Scores generated for hiring decision

---

## 📊 Match Score Logic

- 7–10 → Strong Match ✅
- 4–6 → Moderate Match ⚠
- 0–3 → Weak Match ❌

---

## 🔐 Security

- API keys secured using Streamlit Secrets
- No sensitive data stored

---

## 💡 Future Improvements

- Add authentication system
- Add skill extraction visualization
- Add downloadable PDF report
- Add analytics dashboard

---

