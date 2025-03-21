import streamlit as st
import faiss
import numpy as np
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage as UserMessage
from bs4 import BeautifulSoup
import requests

# Your Mistral API key
MISTRAL_API_KEY = "WXKRllxKbp3htHMI9bogmeOTP0VV09ce"

# Initialize Mistral Client
mistral_client = MistralClient(api_key=MISTRAL_API_KEY)

# Get text embeddings
def generate_embeddings(text_list):
    response = mistral_client.embeddings.create(
        model="mistral-embed",
        inputs=text_list
    )
    return np.array([emb.embedding for emb in response.data])

# Query Mistral LLM
def chat_with_mistral(context, user_query):
    messages = [
        {"role": "user", "content": f"Context: {context}\nQuestion: {user_query}\nAnswer:"}
    ]
    completion = mistral_client.chat(
        model="mistral-large-latest",
        messages=messages
    )
    return completion.choices[0].message.content

# Fetch and parse policy text
def fetch_policy(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    return soup.get_text()

# Create FAISS index
def create_faiss_index(chunks):
    embeddings = generate_embeddings(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Retrieve top relevant text chunks
def retrieve_chunks(query, index, chunks, k=3):
    query_embedding = generate_embeddings([query])
    _, indices = index.search(query_embedding, k)
    return [chunks[idx] for idx in indices[0]]

# Streamlit UI setup
st.title("📚 UDST Policy ChatBot 🤖")

policy_links = {
    "Student Conduct Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-conduct-policy",
    "Academic Schedule Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/academic-schedule-policy",
    "Student Attendance Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-attendance-policy",
    "Student Appeals Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-appeals-policy",
    "Graduation Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduation-policy",
    "Academic Standing Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-standing-policy",
    "Transfer Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/transfer-policy",
    "Admissions Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/admissions-policy",
    "Final Grade Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/final-grade-policy",
    "Registration Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/registration-policy",
}

selected_policy = st.selectbox("Select a Policy:", options=list(policy_links.keys()))

if st.button("Load Policy"):
    with st.spinner('Loading policy and preparing embeddings...'):
        policy_text = fetch_policy(policy_links[selected_policy])
        policy_chunks = [policy_text[i:i+500] for i in range(0, len(policy_text), 500)]
        st.session_state['faiss_index'] = create_faiss_index(policy_chunks)
        st.session_state['policy_chunks'] = policy_chunks
    st.success('Policy loaded successfully!')

user_question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if 'faiss_index' in st.session_state:
        relevant_chunks = retrieve_chunks(user_question, st.session_state['faiss_index'], st.session_state['policy_chunks'])
        context = "\n".join(relevant_chunks)
        answer = chat_with_mistral(context, user_question)
        st.text_area("Answer:", value=answer, height=200)
    else:
        st.error('Please load a policy first.')
