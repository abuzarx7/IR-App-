import streamlit as st
import nltk
from nltk.corpus import reuters
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# NLTK Resource Setup
def setup_nltk_resources():
    nltk.download('reuters')  # Ensure Reuters corpus is downloaded
    try:
        # Check if the Reuters corpus is accessible
        nltk.data.find('corpora/reuters.zip')
    except LookupError:
        st.error("Reuters corpus is not found, please check the download and file path.")

setup_nltk_resources()

# Load and prepare the corpus
corpus_sentences = []
if hasattr(reuters, 'fileids'):
    for fileid in reuters.fileids():
        raw_text = reuters.raw(fileid)
        tokenized_sentence = [word for word in nltk.word_tokenize(raw_text) if word.isalnum()]
        corpus_sentences.append(tokenized_sentence)

# Model training
model = Word2Vec(sentences=corpus_sentences, vector_size=100, window=5, min_count=5, workers=4)

# t-SNE Visualization
def tsne_plot(model):
    labels = []
    tokens = []

    for word in model.wv.index_to_key[:200]:  # Limit to top 200 words for better visualization
        tokens.append(model.wv[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    
    plt.figure(figsize=(10, 10)) 
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

# Define a dummy search function (replace with actual implementation)
def search_function(query):
    # Example mock search results updated
    results = [
        {"id": "training/10576", "score": 0.8033, "preview": "Irwin Jacobs involved in major stock deal, according to sources."},
        {"id": "training/13462", "score": 0.7407, "preview": "Trade battles continue as cocoa buffer stock struggles."},
        {"id": "test/20039", "score": 0.7107, "preview": "BP set to launch new offshore oil project despite environmental concerns."},
    ]
    return results

def display_search_results(results):
    for result in results:
        with st.container():
            st.text(f"Document ID: {result['id']}")
            st.text(f"Similarity Score: {result['score']:.4f}")
            st.text(f"Document Preview: {result['preview']}")
            st.markdown("---")  # Adds a horizontal line for separating entries

st.title('Document Search App')

query = st.text_input("Enter your search query:", "")
if st.button("Search"):
    if query:
        results = search_function(query)
        if results:
            display_search_results(results)
        else:
            st.write("No results found.")
    else:
        st.warning("Please enter a search query.")

# Optionally display the t-SNE plot
if st.button('Show t-SNE Plot'):
    tsne_plot(model)
