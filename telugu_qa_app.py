import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

st.set_page_config(page_title="Telugu QA + Summarizer", page_icon="🧠")

# Title and description
st.title("🧠 Telugu QA + Summarizer App")
st.write("Ask questions from 100 Telugu news articles, view summaries, and explore knowledge interactively!")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("100_articles.csv")
    df['content'] = df['content'].astype(str)
    return df

df = load_data()

# Preprocess text
def split_telugu_sentences(text):
    text = str(text).strip().replace('\n', ' ')
    return re.split(r'(?<=[।.!?])\s+', text)

def preprocess_text(text):
    sentences = split_telugu_sentences(text)
    return " ".join(sentences)

df['content_clean'] = df['content'].apply(preprocess_text)

# Split into 300-word chunks
def split_into_chunks(text, max_words=300):
    words = text.split()
    return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

chunks = []
for content in df['content_clean']:
    chunks.extend(split_into_chunks(content))

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=50000)
chunk_vectors = vectorizer.fit_transform(chunks)

# Load QA pipeline
@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="deepset/xlm-roberta-large-squad2", tokenizer="deepset/xlm-roberta-large-squad2")

qa_pipeline = load_qa_model()

# Answer a question
def retrieve_top_passages(question, top_k=5):
    question_vec = vectorizer.transform([question])
    similarity_scores = cosine_similarity(question_vec, chunk_vectors)[0]
    top_indices = np.argsort(similarity_scores)[-top_k:][::-1]
    return [(chunks[i], similarity_scores[i]) for i in top_indices]

def answer_question(question, top_k=5):
    top_contexts = retrieve_top_passages(question, top_k=top_k)
    best_answer = None
    best_score = -1

    for context, score in top_contexts:
        result = qa_pipeline(question=question, context=context)
        if result['score'] > best_score:
            best_score = result['score']
            best_answer = result['answer']

    return best_answer if best_answer else "❓ Unable to find an answer."

# UI for Q&A
question = st.text_input("🧠 Ask a question from the Telugu articles:")
if question:
    with st.spinner("Thinking..."):
        answer = answer_question(question)
        st.success(f"✅ Answer: {answer}")
