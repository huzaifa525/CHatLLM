import streamlit as st
import PyPDF2
import docx
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import io
from typing import List, Tuple
import os

class DocumentProcessor:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return ""
    
    def extract_text_from_docx(self, docx_file) -> str:
        try:
            doc = docx.Document(docx_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error processing DOCX: {str(e)}")
            return ""
    
    def split_into_chunks(self, text: str, chunk_size: int = 200) -> List[str]:
        """Split text into chunks based on character count"""
        if not text.strip():
            return []
            
        try:
            # Split text into paragraphs
            paragraphs = text.split('\n')
            chunks = []
            current_chunk = ""
            
            for paragraph in paragraphs:
                if len(current_chunk) + len(paragraph) <= chunk_size:
                    current_chunk += paragraph + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph + " "
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Handle case where chunks are too large
            final_chunks = []
            for chunk in chunks:
                if len(chunk) > chunk_size:
                    # Split into smaller chunks based on periods
                    sentences = chunk.split('.')
                    temp_chunk = ""
                    for sentence in sentences:
                        if len(temp_chunk) + len(sentence) <= chunk_size:
                            temp_chunk += sentence + ". "
                        else:
                            if temp_chunk:
                                final_chunks.append(temp_chunk.strip())
                            temp_chunk = sentence + ". "
                    if temp_chunk:
                        final_chunks.append(temp_chunk.strip())
                else:
                    final_chunks.append(chunk)
            
            return final_chunks
        except Exception as e:
            st.error(f"Error splitting text into chunks: {str(e)}")
            return [text]  # Return original text as single chunk if splitting fails
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        try:
            return self.model.encode(texts)
        except Exception as e:
            st.error(f"Error generating embeddings: {str(e)}")
            return np.array([])
    
    def find_most_similar_chunks(self, 
                               query: str, 
                               chunks: List[str], 
                               chunk_embeddings: np.ndarray,
                               top_k: int = 3) -> List[Tuple[str, float]]:
        if not chunks or not query:
            return []
            
        try:
            query_embedding = self.model.encode([query])
            similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
            
            most_similar_indices = similarities.argsort()[-top_k:][::-1]
            results = []
            
            for idx in most_similar_indices:
                results.append((chunks[idx], similarities[idx]))
            
            return results
        except Exception as e:
            st.error(f"Error finding similar chunks: {str(e)}")
            return []

@st.cache_resource
def get_document_processor():
    return DocumentProcessor()

def process_documents(files, doc_processor):
    all_text = ""
    progress_bar = st.progress(0)
    
    for i, file in enumerate(files):
        progress = (i + 1) / len(files)
        progress_bar.progress(progress)
        
        if file.type == "application/pdf":
            all_text += doc_processor.extract_text_from_pdf(file)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            all_text += doc_processor.extract_text_from_docx(file)
            
    progress_bar.empty()
    return all_text

def main():
    st.set_page_config(page_title="Document QA without LLM", layout="wide")
    st.title("Document QA without LLM")
    st.write("Upload your documents and ask questions!")
    
    # Initialize the document processor
    doc_processor = get_document_processor()
    
    # Session state for storing processed data
    if 'chunks' not in st.session_state:
        st.session_state.chunks = None
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
    
    # File upload
    uploaded_files = st.file_uploader("Upload your documents (PDF/DOCX)", 
                                    type=["pdf", "docx"],
                                    accept_multiple_files=True)
    
    if uploaded_files:
        # Process documents
        all_text = process_documents(uploaded_files, doc_processor)
        
        if all_text.strip():
            # Split text into chunks
            if st.session_state.chunks is None:
                st.session_state.chunks = doc_processor.split_into_chunks(all_text)
            
            if st.session_state.chunks:
                # Get embeddings for all chunks
                if st.session_state.embeddings is None:
                    with st.spinner("Processing documents..."):
                        st.session_state.embeddings = doc_processor.get_embeddings(st.session_state.chunks)
                
                if len(st.session_state.embeddings) > 0:
                    st.success("Documents processed successfully!")
                    
                    # Query input
                    query = st.text_input("Ask a question about your documents:")
                    
                    if query:
                        with st.spinner("Searching for relevant information..."):
                            similar_chunks = doc_processor.find_most_similar_chunks(
                                query, 
                                st.session_state.chunks, 
                                st.session_state.embeddings
                            )
                        
                        if similar_chunks:
                            st.subheader("Most Relevant Information:")
                            for chunk, similarity in similar_chunks:
                                with st.container():
                                    st.markdown("---")
                                    st.markdown(f"**Relevance Score:** {similarity:.2f}")
                                    st.write(chunk)
                        else:
                            st.warning("No relevant information found.")
                else:
                    st.error("Error processing document embeddings.")
            else:
                st.error("Error processing document text.")
        else:
            st.error("No text could be extracted from the uploaded documents.")

    # Add clear button
    if st.button("Clear All"):
        st.session_state.chunks = None
        st.session_state.embeddings = None
        st.experimental_rerun()

if __name__ == "__main__":
    main()

# Requirements (save as requirements.txt):
# streamlit
# PyPDF2
# python-docx
# sentence-transformers
# scikit-learn
# torch
# numpy
