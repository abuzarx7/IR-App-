import streamlit as st
import nltk
from nltk.corpus import reuters, stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
nltk.download('reuters')
nltk.download('punkt')
nltk.download('stopwords')

# Preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    return [word.lower() for word in word_tokenize(text) if word.isalnum() and word.lower() not in stop_words]

# Get average embedding for a list of tokens
def get_average_embedding(tokens, model):
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# Compute document embeddings for the corpus
@st.cache_data  # Cache the computation to avoid re-running on every interaction
def compute_document_embeddings(corpus_sentences, model):
    document_embeddings = []
    for sentence in corpus_sentences:
        embedding = get_average_embedding(sentence, model)
        document_embeddings.append(embedding)
    return document_embeddings

# Find top N documents based on query
def find_top_n_documents(query, document_embeddings, corpus_sentences, model, N=5):
    query_tokens = preprocess_text(query)
    query_embedding = get_average_embedding(query_tokens, model)
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    top_indices = np.argsort(similarities)[-N:][::-1]
    return [(reuters.fileids()[idx], similarities[idx], corpus_sentences[idx][:10]) for idx in top_indices]

# Load and preprocess the Reuters corpus
@st.cache_data
def load_corpus():
    corpus_sentences = []
    for fileid in reuters.fileids():
        raw_text = reuters.raw(fileid)
        corpus_sentences.append(preprocess_text(raw_text))
    return corpus_sentences

# Train Word2Vec model
@st.cache_resource
def train_word2vec(corpus_sentences):
    model = Word2Vec(sentences=corpus_sentences, vector_size=100, window=5, min_count=5, workers=4)
    return model

# Main Streamlit app
def main():
    st.title("Information Retrieval with Word2Vec")
    st.write("Enter a query to find the top relevant documents from the Reuters corpus.")

    # Load corpus and train model
    with st.spinner("Loading corpus and training model..."):
        corpus_sentences = load_corpus()
        model = train_word2vec(corpus_sentences)
        document_embeddings = compute_document_embeddings(corpus_sentences, model)

    # User input
    query = st.text_input("Enter your query (e.g., 'stock market performance'):", "stock market performance")
    N = st.slider("Number of top documents to retrieve:", 1, 10, 5)

    if st.button("Search"):
        with st.spinner("Searching for relevant documents..."):
            results = find_top_n_documents(query, document_embeddings, corpus_sentences, model, N)
        
        st.subheader(f"Top {N} Most Relevant Documents for Query: '{query}'")
        for doc_id, similarity, preview in results:
            preview_text = ' '.join(preview) + "..."
            st.write(f"**Document ID**: {doc_id}")
            st.write(f"**Similarity Score**: {similarity:.4f}")
            st.write(f"**Preview**: {preview_text}")
            st.write("---")

if __name__ == "__main__":
    main()