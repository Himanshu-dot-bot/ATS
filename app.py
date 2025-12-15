import streamlit as st # type: ignore
import base64
import docx2txt # type: ignore
import re
import nltk  # type: ignore
import fitz  # PyMuPDF  # type: ignore
import io
import pandas as pd  # type: ignore
import plotly.express as px # type: ignore
from nltk.corpus import wordnet # type: ignore
from functools import lru_cache

# --------------------------------------
# Setup
# --------------------------------------
st.set_page_config(page_title="Advanced ATS Resume Checker", page_icon="📄", layout="wide")
st.title("📄 Advanced ATS Analyzer & Ranker 🚀")
nltk.download('wordnet', quiet=True)

# --------------------------------------
# Helper Functions
# --------------------------------------
# Text Extraction
def get_text_from_file(file):
    text = ""
    if not file:
        return text
    file_type = file.name.split(".")[-1].lower()
    file.seek(0)
    if file_type == "pdf":
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        for page in pdf:
            text += page.get_text() + "\n"
        pdf.close()
    elif file_type == "docx":
        text = docx2txt.process(file)
    file.seek(0)
    return text

# Clean and tokenize
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text

def extract_keywords(text):
    stopwords = {"and","or","the","a","an","in","on","with","for","to","of","is","are"}
    words = re.findall(r'\b\w+\b', clean_text(text))
    return set(w for w in words if w not in stopwords)

# Synonyms with caching
@lru_cache(maxsize=500)
def get_synonyms(word):
    syns = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            syns.add(lemma.name().replace("_", " ").lower())
    return syns

# ATS Calculation
def calculate_ats(resume_text, jd_text):
    resume_kw = extract_keywords(resume_text)
    jd_kw = extract_keywords(jd_text)

    matches = set()
    missing = set()

    for word in jd_kw:
        if word in resume_kw:
            matches.add(word)
        elif get_synonyms(word).intersection(resume_kw):
            matches.add(word)
        else:
            missing.add(word)

    ats = int((len(matches) / len(jd_kw)) * 100) if jd_kw else 0
    priority = {"python", "java", "sql", "aws", "react", "node", "mongodb", "docker"}
    weighted = min(ats + len(matches.intersection(priority)) * 2, 100)
    
    return ats, weighted, matches, missing

# PDF Highlight
def highlight_pdf_keywords(file, keywords):
    file.seek(0)
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    for page in pdf:
        for kw in keywords:
            try:
                # Exact search first
                found = page.search_for(kw)
                for inst in found:
                    page.add_highlight_annot(inst)
                # Case-insensitive fallback
                words = page.get_text("words")
                for inst in words:
                    x0, y0, x1, y1, word = inst[:5]
                    if word.lower() == kw.lower():
                        rect = fitz.Rect(x0, y0, x1, y1)
                        page.add_highlight_annot(rect)
            except Exception as e:
                print(f"Error highlighting {kw}: {e}")
    pdf_bytes = pdf.write()
    pdf.close()
    return pdf_bytes

def show_pdf_bytes(pdf_bytes):
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    iframe = f'<iframe src="data:application/pdf;base64,{b64}" width="800" height="1000"></iframe>'
    st.markdown(iframe, unsafe_allow_html=True)

def make_pdf_download_link(name, file_bytes):
    b64 = base64.b64encode(file_bytes).decode()
    return f'<a href="data:application/pdf;base64,{b64}" download="{name}">📄 Open {name}</a>'

# --------------------------------------
# SINGLE RESUME ATS
# --------------------------------------
st.header("📌 Single Resume ATS Checker")

resume_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"], key="single")
jd = st.text_area("Paste Job Description", height=150, key="jd_single")

if st.button("Analyze Resume"):
    if resume_file and jd.strip():
        text = get_text_from_file(resume_file)
        ats, weighted, matches, missing = calculate_ats(text, jd)

        st.subheader("📄 Highlighted PDF Preview")
        if resume_file.name.endswith(".pdf"):
            pdf_bytes = highlight_pdf_keywords(resume_file, matches)
            show_pdf_bytes(pdf_bytes)

        st.write(f"**ATS Score:** {ats}%")
        st.progress(ats)
        st.write(f"**Weighted Score:** {weighted}%")

        st.subheader("Matched Keywords")
        st.write(", ".join(matches) if matches else "None")

        st.subheader("Missing Keywords")
        st.write(", ".join(missing) if missing else "None")

# --------------------------------------
# MULTI-SCAN RANKING
# --------------------------------------
st.header("📊 Multiple Resume Comparison & Ranking")

num = st.number_input("How many resumes to compare?", 2, 50, 5)
files = st.file_uploader("Upload All Resumes", type=["pdf", "docx"], accept_multiple_files=True)
jd_multi = st.text_area("Paste Job Description for Ranking", height=150, key="jd_multi")

if st.button("🔍 Analyze All Resumes"):
    if files and jd_multi.strip():
        results = []
        pdf_links = {}

        for f in files:
            txt = get_text_from_file(f)
            ats, weighted, matches, _ = calculate_ats(txt, jd_multi)

            if f.name.endswith(".pdf"):
                pdf_bytes = highlight_pdf_keywords(f, matches)
                pdf_links[f.name] = pdf_bytes

            results.append({
                "Resume": f.name,
                "ATS Score": ats,
                "Weighted Score": weighted,
                "Matched Keywords": ", ".join(matches)
            })

        df = pd.DataFrame(results).sort_values("Weighted Score", ascending=False)
        df.reset_index(drop=True, inplace=True)
        df.index += 1

        st.subheader("🏆 Ranking Table")
        st.dataframe(df)

        fig = px.bar(
            df,
            x=df.index,
            y="Weighted Score",
            text="Weighted Score",
            labels={"x": "Rank", "Weighted Score": "Score"},
            title="📊 Ranking Bar Graph"
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📄 Open Highlighted PDFs")
        for name, pdf_bytes in pdf_links.items():
            st.markdown(make_pdf_download_link(name, pdf_bytes), unsafe_allow_html=True)

# --------------------------------------
# CLEAR BUTTON
# --------------------------------------
if st.button("♻ Reset / Clear All"):
    st.session_state.clear()
    st.rerun()
