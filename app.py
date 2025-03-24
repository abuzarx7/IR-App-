import streamlit as st
import os
import nltk
from nltk.corpus import reuters, stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Setup NLTK downloads
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

for resource in ["reuters", "punkt", "stopwords"]:
    try:
        nltk.data.find(f"corpora/{resource}" if resource != "punkt" else "tokenizers/punkt")
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_dir, quiet=True)

st.set_page_config(page_title="Word2Vec IR App", layout="wide")
st.title("📚 Information Retrieval using Word2Vec on Reuters Corpus")
st.markdown("""
This app uses **NLTK Reuters Corpus** to train a **Word2Vec** model and visualize the word embeddings using **t-SNE**.
""")

# Avoid caching tokenizer-dependent functions
def load_corpus():
    corpus_sentences = []
    for fileid in reuters.fileids():
        raw_text = reuters.raw(fileid)
        tokenized = [word for word in word_tokenize(raw_text) if word.isalnum()]
        corpus_sentences.append(tokenized)
    return corpus_sentences

corpus_sentences = load_corpus()
st.success(f"Loaded {len(corpus_sentences)} documents from Reuters corpus.")

@st.cache_resource
def train_model(sentences):
    model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=5, workers=4)
    return model

model = train_model(corpus_sentences)
vocab = list(model.wv.index_to_key)
st.info(f"Trained Word2Vec model with a vocabulary size of {len(vocab)} words.")

# Word similarity section
st.header("🔍 Word Similarity")
word = st.text_input("Enter a word from the vocabulary:", "oil")
if word in model.wv:
    similar_words = model.wv.most_similar(word, topn=10)
    st.write("Top similar words:")
    st.dataframe(pd.DataFrame(similar_words, columns=["Word", "Similarity"]))
else:
    st.warning("Word not in vocabulary!")

# t-SNE visualization
st.header("🧠 Word Embedding Visualization (t-SNE)")
num_words = st.slider("Number of words to visualize:", 100, 500, 200)
selected_words = vocab[:num_words]
vectors = np.array([model.wv[word] for word in selected_words])

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
reduced_vectors = tsne.fit_transform(vectors)

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1])
for i, word in enumerate(selected_words):
    ax.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]), fontsize=8)
st.pyplot(fig)

# Dummy search functionality
def search_function(query):
    return [
        {"id": "training/10576", "score": 0.8033, "preview": "Sample document preview 1..."},
        {"id": "training/13462", "score": 0.7444, "preview": "Sample document preview 2..."},
    ]

st.header("🔎 Search the Corpus")
query = st.text_input("Enter your search query:")
if query:
    st.write(f"Searching for: {query}")
    results = search_function(query)
    for result in results:
        st.markdown(f"### 📌 {result['id']}")
        st.write(f"**Similarity Score:** {result['score']:.4f}")
        st.text_area("Document Excerpt:", result['preview'], height=100, disabled=True)
        st.markdown("---")
