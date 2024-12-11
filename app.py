import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import faiss
import os

# ----------------------------
# Setup
# ----------------------------
# Load embedding model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load QA model
qa_model_name = "distilbert-base-uncased-distilled-squad"
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)

# ----------------------------
# Document Preparation
# ----------------------------
# Suppose you have a folder 'docs' with text files, or a single large text.
# We'll load them all, split by paragraphs, and create embeddings.
def load_documents(doc_folder='docs'):
    docs = []
    for file in os.listdir(doc_folder):
        if file.endswith(".txt"):
            with open(os.path.join(doc_folder, file), 'r', encoding='utf-8') as f:
                content = f.read()
                paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
                docs.extend(paragraphs)
    return docs

@st.cache_data
def build_index():
    docs = load_documents()
    # Encode all docs
    doc_embeddings = embedding_model.encode(docs, show_progress_bar=True)
    doc_embeddings = doc_embeddings.astype('float32')
    
    # Create FAISS index
    index = faiss.IndexFlatIP(doc_embeddings.shape[1]) # cosine similarity
    index.add(doc_embeddings)
    return docs, index

docs, index = build_index()

# ----------------------------
# Retrieval Function
# ----------------------------
def retrieve_docs(query, top_k=3):
    query_embedding = embedding_model.encode([query]).astype('float32')
    # Search in FAISS
    scores, indices = index.search(query_embedding, top_k)
    retrieved = [(docs[i], scores[0][j]) for j, i in enumerate(indices[0])]
    return retrieved

# ----------------------------
# QA Function
# ----------------------------
def answer_question(question, context):
    inputs = qa_tokenizer.encode_plus(question, context, return_tensors='pt')
    with torch.no_grad():
        outputs = qa_model(**inputs)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    
    # Get the most likely start and end of answer tokens
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores) + 1
    answer = qa_tokenizer.convert_tokens_to_string(qa_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    return answer.strip()

# ----------------------------
# Streamlit App
# ----------------------------
st.title("RAG-Style Document QA (No LLM)")

user_query = st.text_input("Ask a question about the documents:")
if user_query:
    with st.spinner("Searching and answering..."):
        retrieved = retrieve_docs(user_query, top_k=3)
        # Use the top document for QA (you can combine or iterate through all)
        best_context, score = retrieved[0]
        final_answer = answer_question(user_query, best_context)

    st.subheader("Answer:")
    st.write(final_answer)

    st.subheader("Top Retrieved Passages:")
    for i, (passage, sc) in enumerate(retrieved):
        st.write(f"**Passage {i+1} (score: {sc:.4f})**:\n{passage}")
