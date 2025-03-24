import streamlit as st
import nltk
from nltk.corpus import reuters, stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

# Download NLTK data
nltk.download('reuters')
nltk.download('punkt')
nltk.download('stopwords')

# Load and preprocess Reuters corpus
corpus_sentences = []
document_store = {}

for fileid in reuters.fileids():
    raw_text = reuters.raw(fileid)
    document_store[fileid] = raw_text
    corpus_sentences.append(raw_text)

# Vectorize using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(corpus_sentences)
doc_ids = list(document_store.keys())

# Streamlit UI
st.title("📚 IR App Search Engine")
st.write("Enter a search query to find relevant documents from the Reuters corpus:")

query = st.text_input("Enter your search query:", "")

if query:
    query_vec = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    results = []
    for idx, score in enumerate(cosine_similarities):
        if score > 0:  # only show relevant documents
            doc_id = doc_ids[idx]
            preview = document_store[doc_id][:300].replace('\n', ' ') + "..."
            results.append({
                "id": doc_id,
                "score": score,
                "preview": preview
            })
    
    # Sort results by similarity score and keep top 5 only
    results = sorted(results, key=lambda x: x["score"], reverse=True)[:5]

    # Display results
    for result in results:
        st.subheader(f"📌 {result['id']}")
        st.write(f"**Similarity Score:** {result['score']:.4f}")
        st.text_area("Document Excerpt:", result["preview"], height=120)
