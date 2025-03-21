import streamlit as st
import faiss
import numpy as np
from mistralai import Mistral, UserMessage
from bs4 import BeautifulSoup
import requests

# Mistral API key
MISTRAL_API_KEY = "WXKRllxKbp3htHMI9bogmeOTP0VV09ce"

# Initialize Mistral Client
client = Mistral(api_key=MISTRAL_API_KEY)

# Function to extract text from policy URLs
def extract_policy_text(policy_url):
    html = requests.get(policy_url).content
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text()

# Generate text embeddings
def create_embeddings(text_list):
    response = client.embeddings.create(model="mistral-embed", inputs=text_list)
    return np.array([r.embedding for r in response.data])

# FAISS index creation
def setup_faiss_index(policy_text):
    chunks = [policy_text[i:i+500] for i in range(0, len(policy_text), 500)]
    embeddings = create_embeddings(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks

# Retrieve chunks based on user query
def fetch_relevant_chunks(query, index, chunks, num_chunks=2):
    query_embedding = create_embeddings([query])
    _, indices = index.search(query_embedding, num_chunks)
    return [chunks[idx] for idx in indices[0]]

# Chat with Mistral AI
def ask_mistral(context_chunks, query):
    context = "\n".join(context_chunks)
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[UserMessage(content=prompt)]
    )
    return response.choices[0].message.content

# Streamlit User Interface
st.title("📖 UDST Interactive Policy Assistant 🤖")

# UDST policies
udst_policies = {
    "Student Conduct": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-conduct-policy",
    "Academic Schedule": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/academic-schedule-policy",
    "Attendance": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-attendance-policy",
    "Appeals": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-appeals-policy",
    "Graduation": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduation-policy",
    "Academic Standing": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-standing-policy",
    "Transfer": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/transfer-policy",
    "Admissions": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/admissions-policy",
    "Final Grade": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/final-grade-policy",
    "Registration": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/registration-policy",
}

selected_policy = st.selectbox("Choose Policy:", list(udst_policies.keys()))

# Load policy button
if st.button("Load Selected Policy"):
    with st.spinner('Preparing policy content...'):
        policy_text = extract_policy_text(udst_policies[selected_policy])
        index, policy_chunks = setup_faiss_index(policy_text)
        st.session_state['faiss_index'] = index
        st.session_state['policy_chunks'] = policy_chunks
    st.success("Policy loaded successfully!")

# User query input
user_query = st.text_input("Enter your question:")

# Fetch answer button
if st.button("Get Policy Answer"):
    if 'faiss_index' in st.session_state:
        relevant_context = fetch_relevant_chunks(user_query, st.session_state['faiss_index'], st.session_state['policy_chunks'])
        answer = ask_mistral(relevant_context, user_query)
        st.text_area("Answer:", value=answer, height=220)
    else:
        st.warning("Please load a policy first!")
