import streamlit as st
import PyPDF2
import docx
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import io
from typing import List, Tuple

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class DocumentProcessor:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
    def extract_text_from_docx(self, docx_file) -> str:
        doc = docx.Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def split_into_chunks(self, text: str, chunk_size: int = 3) -> List[str]:
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            current_chunk.append(sentence)
            current_length += 1
            
            if current_length >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts)
    
    def find_most_similar_chunks(self, 
                               query: str, 
                               chunks: List[str], 
                               chunk_embeddings: np.ndarray,
                               top_k: int = 3) -> List[Tuple[str, float]]:
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
        
        most_similar_indices = similarities.argsort()[-top_k:][::-1]
        results = []
        
        for idx in most_similar_indices:
            results.append((chunks[idx], similarities[idx]))
        
        return results

def main():
    st.title("Document QA without LLM")
    st.write("Upload your documents and ask questions!")
    
    # Initialize the document processor
    doc_processor = DocumentProcessor()
    
    # File upload
    uploaded_files = st.file_uploader("Upload your documents (PDF/DOCX)", 
                                    type=["pdf", "docx"],
                                    accept_multiple_files=True)
    
    if uploaded_files:
        # Process documents
        all_text = ""
        for file in uploaded_files:
            if file.type == "application/pdf":
                all_text += doc_processor.extract_text_from_pdf(file)
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                all_text += doc_processor.extract_text_from_docx(file)
        
        # Split text into chunks
        chunks = doc_processor.split_into_chunks(all_text)
        
        # Get embeddings for all chunks
        with st.spinner("Processing documents..."):
            chunk_embeddings = doc_processor.get_embeddings(chunks)
        
        st.success("Documents processed successfully!")
        
        # Query input
        query = st.text_input("Ask a question about your documents:")
        
        if query:
            with st.spinner("Searching for relevant information..."):
                similar_chunks = doc_processor.find_most_similar_chunks(
                    query, chunks, chunk_embeddings
                )
            
            st.subheader("Most Relevant Information:")
            for chunk, similarity in similar_chunks:
                st.write("---")
                st.write(chunk)
                st.write(f"Relevance Score: {similarity:.2f}")

if __name__ == "__main__":
    main()
