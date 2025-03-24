import streamlit as st

# Define a dummy search function (replace with actual implementation)
def search_function(query):
    # Example mock search results
    results = [
        {"id": "training/10576", "score": 0.8033, "preview": "Sample document preview 1..."},
        {"id": "training/13462", "score": 0.7444, "preview": "Sample document preview 2..."},
    ]
    return results

import streamlit as st
#!/usr/bin/env python
# coding: utf-8

# In[52]:

import streamlit as st

import nltk
from nltk.corpus import reuters, stopwords
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd


# In[4]:


nltk.download('reuters')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


# In[7]:


corpus_sentences = []
for fileid in reuters.fileids():
    raw_text = reuters.raw(fileid)
    tokenized_sentence = [word for word in nltk.word_tokenize(raw_text) if word.isalnum() and word]
    corpus_sentences.append(tokenized_sentence)
st.write(f"Number of sentences in the Reuters corpus: {len(corpus_sentences)}")


# In[9]:


model = Word2Vec(sentences=corpus_sentences, vector_size=100, window=5, min_count=5, workers=4)
# Print vocabulary size
st.write(f"Vocabulary size: {len(model.wv.index_to_key)}")


# In[11]:


import numpy as np
# Extract the learned word vectors and their corresponding words for visualization.
words = list(model.wv.index_to_key)[:200]  # Limit to top 200 words for better visualization
word_vectors = np.array([model.wv[word] for word in words])  # Convert to NumPy array for compatible 


# In[13]:


# Use t-SNE to project the high-dimensional word embeddings into a 2D space.
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
word_vectors_2d = tsne.fit_transform(word_vectors)


# In[15]:


# Plot the 2D t-SNE visualization of the word embeddings with their labels.
def plot_embeddings(vectors, labels):
    plt.figure(figsize=(16, 12))
    for i, label in enumerate(labels):
        x, y = vectors[i]
        plt.scatter(x, y, color='blue')
        plt.text(x + 0.1, y + 0.1, label, fontsize=9)
if st.checkbox('Show visualization'):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    # Ensure vectors[i] is defined before use
    x, y = [1, 2, 3], [1, 2, 3]  # Replace with actual data
    ax.scatter(x, y, color='blue')
    ax.text(x[0] + 0.1, y[0] + 0.1, 'Sample', fontsize=9)
    st.pyplot(fig)
query = st.text_input('Enter your search query:', '')
if query:
    st.write(f'Searching for: {query}')
    search_results = search_function(query)  # Replace with actual function
    for result in search_results:
        st.markdown(f'### 📌 {result["id"]}')
        st.write(f'**Similarity Score:** {result["score"]:.4f}')
        preview_text = result['preview'].strip() if result['preview'] else '⚠️ No preview available for this document.'
        st.text_area('Document Excerpt:', preview_text, height=100, disabled=True)
        st.write(f'📖 *Why this document?* This document contains terms relevant to **"{query}"**.')
        st.markdown('---')
        st.write(f'Document ID: {result["id"]}')
        st.write(f'Similarity Score: {result["score"]}')
        st.write(f'Document Preview: {result["preview"]}')
        st.write('---')
