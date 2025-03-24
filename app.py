
import streamlit as st
import nltk
from nltk.corpus import reuters, stopwords
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

nltk.download('reuters')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

corpus_sentences = []
for fileid in reuters.fileids():
    raw_text = reuters.raw(fileid)
    tokenized_sentence = [word for word in nltk.word_tokenize(raw_text) if word.isalnum() and word]
    corpus_sentences.append(tokenized_sentence)
st.write(f"Number of sentences in the Reuters corpus: {len(corpus_sentences)}")

model = Word2Vec(sentences=corpus_sentences, vector_size=100, window=5, min_count=5, workers=4)
st.write(f"Vocabulary size: {len(model.wv.index_to_key)}")

# Placeholder for your search function. This should interface with your actual search backend.
def search_function(query):
    # Example results structure
    results = [
        {"id": "training/10576", "score": 0.8033, "preview": "Amid rumors Irwin Jacobs sold stock arbitrageurs say amid rumors..."},
        {"id": "training/13462", "score": 0.7407, "preview": "Cocoa buffer stock may face uphill battle trade international cocoa..."},
        {"id": "test/20039", "score": 0.7107, "preview": "Lawson says BP share offer going ahead Chancellor Exchequer Nigel..."},
        {"id": "test/17499", "score": 0.7089, "preview": "Dutch cocoa processors unhappy ICCO buffer Dutch cocoa processors unhappy..."},
        {"id": "test/18014", "score": 0.7076, "preview": "ICCO buys tonnes cocoa buffer stock International Cocoa Organization ICCO..."}
    ]
    return results

# Streamlit application layout
st.title("Document Search Engine")
query = st.text_input("Enter your search query:", "stock market performance")

if st.button("Search"):
    results = search_function(query)
    for result in results:
        st.subheader(f"Document ID: {result['id']}")
        st.write(f"Similarity Score: {result['score']}")
        st.write(f"Document Preview: {result['preview']}")
        st.write("---")  # Adds a line separator between entries
