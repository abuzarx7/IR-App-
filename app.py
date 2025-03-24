import streamlit as st
import os
import nltk
from nltk.corpus import reuters
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Setup for NLTK on Streamlit Cloud
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download required corpora
nltk.download("reuters", download_dir=nltk_data_dir)
nltk.download("punkt", download_dir=nltk_data_dir)
nltk.download("stopwords", download_dir=nltk_data_dir)

# Page config
st.set_page_config(page_title="IR Word2Vec App", layout="wide")
st.title("📘 Word2Vec Information Retrieval App")
st.markdown("""
This app uses the **Reuters corpus** to train a **Word2Vec** model and visualize word embeddings using **t-SNE**.
""")

@st.cache_data
def load_corpus():
    sentences = []
    for fileid in reuters.fileids():
        raw = reuters.raw(fileid)
        tokens = [word for word in word_tokenize(raw) if word.isalnum()]
        sentences.append(tokens)
    return sentences

corpus = load_corpus()
st.success(f"Loaded {len(corpus)} documents from Reuters.")

@st.cache_resource
def train_word2vec(corpus):
    return Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=5, workers=4)

model = train_word2vec(corpus)
vocab = list(model.wv.index_to_key)
st.info(f"Trained Word2Vec model with {len(vocab)} words.")

# Similarity
st.header("🔍 Word Similarity")
input_word = st.text_input("Enter a word to find similar words:", "oil")
if input_word in model.wv:
    sim_words = model.wv.most_similar(input_word, topn=10)
    st.dataframe(pd.DataFrame(sim_words, columns=["Word", "Similarity"]))
else:
    st.warning("Word not found in vocabulary!")

# Visualization
st.header("📊 t-SNE Word Embedding Visualization")
word_count = st.slider("Number of words to visualize:", 100, 500, 200)
chosen_words = vocab[:word_count]
vectors = np.array([model.wv[word] for word in chosen_words])

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
reduced = tsne.fit_transform(vectors)

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(reduced[:, 0], reduced[:, 1])
for i, word in enumerate(chosen_words):
    ax.annotate(word, (reduced[i, 0], reduced[i, 1]), fontsize=8)
st.pyplot(fig)
