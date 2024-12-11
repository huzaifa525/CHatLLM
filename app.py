import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import faiss
import io

# ----------------------------
# Setup
# ----------------------------
st.title("RAG-Style Document QA (No LLM)")

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# Load QA model
@st.cache_resource
def load_qa_model():
    qa_model_name = "distilbert-base-uncased-distilled-squad"
    tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
    return tokenizer, model

qa_tokenizer, qa_model = load_qa_model()

# ----------------------------
# Document and Index Management
# ----------------------------
@st.cache_data
def build_index_from_docs(docs):
    # Encode all docs
    doc_embeddings = embedding_model.encode(docs, show_progress_bar=True)
    doc_embeddings = doc_embeddings.astype('float32')
    
    # Create FAISS index
    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    index.add(doc_embeddings)
    return docs, index

def retrieve_docs(query, docs, index, top_k=3):
    query_embedding = embedding_model.encode([query]).astype('float32')
    scores, indices = index.search(query_embedding, top_k)
    retrieved = [(docs[i], scores[0][j]) for j, i in enumerate(indices[0])]
    return retrieved

def answer_question(question, context):
    inputs = qa_tokenizer.encode_plus(question, context, return_tensors='pt')
    with torch.no_grad():
        outputs = qa_model(**inputs)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores) + 1
    answer = qa_tokenizer.convert_tokens_to_string(
        qa_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    )
    return answer.strip()

# ----------------------------
# Streamlit App Logic
# ----------------------------

st.write("Upload your text documents (as .txt files) and then ask questions!")

uploaded_files = st.file_uploader("Upload text files", type=["txt"], accept_multiple_files=True)

if uploaded_files:
    # Parse uploaded files into docs list
    docs = []
    for uploaded_file in uploaded_files:
        # Read the content of the file
        file_content = uploaded_file.read().decode("utf-8", errors='ignore')
        # Split by paragraphs (or lines)
        paragraphs = [p.strip() for p in file_content.split('\n') if p.strip()]
        docs.extend(paragraphs)

    # Build or rebuild the index
    docs, index = build_index_from_docs(docs)
    
    user_query = st.text_input("Ask a question about the uploaded documents:")
    if user_query and docs:
        with st.spinner("Searching and answering..."):
            retrieved = retrieve_docs(user_query, docs, index, top_k=3)
            best_context, score = retrieved[0]
            final_answer = answer_question(user_query, best_context)

        st.subheader("Answer:")
        st.write(final_answer)

        st.subheader("Top Retrieved Passages:")
        for i, (passage, sc) in enumerate(retrieved):
            st.write(f"**Passage {i+1} (score: {sc:.4f})**:\n{passage}")
