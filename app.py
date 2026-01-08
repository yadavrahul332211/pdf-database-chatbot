import os
import sqlite3
import streamlit as st
from pypdf import PdfReader

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="PDF Database Chatbot", layout="wide")
st.title("🤖 PDF Database Chatbot")


# -------------------------
# LOAD PDF PATHS (CLOUD SAFE)
# -------------------------
def load_pdf_paths():
    pdf_dir = "pdfs"
    return [
        os.path.join(pdf_dir, f)
        for f in os.listdir(pdf_dir)
        if f.endswith(".pdf")
    ]


# -------------------------
# READ PDF
# -------------------------
def read_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


# -------------------------
# BUILD VECTOR DATABASE
# -------------------------
@st.cache_resource
def build_vector_db():
    pdf_paths = load_pdf_paths()

    all_text = ""
    for path in pdf_paths:
        all_text += read_pdf(path)

    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(all_text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = FAISS.from_texts(chunks, embeddings)
    return vector_db


vector_db = build_vector_db()


# -------------------------
# CHATBOT FUNCTION
# -------------------------
def chatbot(question):
    docs = vector_db.similarity_search(question, k=3)
    answer = ""
    for doc in docs:
        answer += doc.page_content + "\n\n"
    return answer


# -------------------------
# UI
# -------------------------
question = st.text_input("💬 Apna question yahan likho:")

if st.button("Ask"):
    if question:
        with st.spinner("Answer dhundh raha hoon..."):
            response = chatbot(question)
        st.success("Answer:")
        st.write(response)
    else:
        st.warning("Please question likho")
