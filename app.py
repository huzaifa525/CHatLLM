import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import faiss
import spacy
import io

# ----------------------------
# Setup
# ----------------------------
st.title("RAG-Style Document QA (No LLM)")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

@st.cache_resource
def load_qa_model():
    # A more capable QA model
    qa_model_name = "deepset/bert-large-uncased-whole-word-masking-squad2"
    tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
    return tokenizer, model

qa_tokenizer, qa_model = load_qa_model()

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()


# ----------------------------
# Document Processing
# ----------------------------
def chunk_text(text, chunk_size=200):
    # Split text into words and chunk
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

def process_documents(uploaded_files, chunk_size):
    docs = []
    for uploaded_file in uploaded_files:
        file_content = uploaded_file.read().decode("utf-8", errors='ignore')
        # Use spaCy to break the document into sentences, then chunk sentences
        doc_spacy = nlp(file_content)
        # Extract text from sentences
        full_text = "\n".join([sent.text.strip() for sent in doc_spacy.sents if sent.text.strip()])
        
        # Chunk the full text
        for ch in chunk_text(full_text, chunk_size=chunk_size):
            docs.append(ch.strip())
    return [d for d in docs if d.strip()]


@st.cache_data(show_spinner=True)
def build_index_from_docs(docs):
    doc_embeddings = embedding_model.encode(docs, show_progress_bar=False).astype('float32')
    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    index.add(doc_embeddings)
    return docs, index


def preprocess_query(query):
    # Use spaCy to normalize query (lemmatization, lowercasing)
    doc = nlp(query)
    processed_tokens = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
    # Join back into a processed query
    processed_query = " ".join(processed_tokens)
    return processed_query


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

    start_idx = torch.argmax(start_scores)
    end_idx = torch.argmax(end_scores) + 1

    all_tokens = qa_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    answer_tokens = all_tokens[start_idx:end_idx]
    answer = qa_tokenizer.convert_tokens_to_string(answer_tokens).strip()
    
    # If the model returns '[CLS]' or empty, it might mean no answer found
    if answer in ["[CLS]", ""]:
        answer = "No clear answer found."
    return answer


# ----------------------------
# Streamlit Interface
# ----------------------------

st.write("Upload your text documents and then ask questions!")
uploaded_files = st.file_uploader("Upload text files", type=["txt"], accept_multiple_files=True)

chunk_size = st.slider("Chunk size (number of words per chunk)", min_value=50, max_value=500, value=200, step=50)

if uploaded_files:
    docs = process_documents(uploaded_files, chunk_size)
    if docs:
        docs, index = build_index_from_docs(docs)

        user_query = st.text_input("Ask a question about the uploaded documents:")
        if user_query and docs:
            with st.spinner("Processing your query..."):
                # Preprocess the query with NLP
                processed_query = preprocess_query(user_query)
                retrieved = retrieve_docs(processed_query, docs, index, top_k=3)
                
                # Use the top retrieved passage for QA
                best_context, score = retrieved[0]
                final_answer = answer_question(user_query, best_context)

            st.subheader("Answer:")
            st.write(final_answer)

            st.subheader("Top Retrieved Passages:")
            for i, (passage, sc) in enumerate(retrieved):
                st.write(f"**Passage {i+1} (score: {sc:.4f})**:\n{passage}")
    else:
        st.write("No documents processed. Make sure your files contain readable text.")
else:
    st.write("Please upload one or more text files to begin.")
