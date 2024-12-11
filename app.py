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
import os

# Create NLTK data directory if it doesn't exist
nltk_data_dir = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Download NLTK data with proper error handling
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

# Initialize NLTK
download_nltk_data()

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
    
    def split_into_chunks(self, text: str, chunk_size: int = 3) -> List[str]:
        if not text.strip():
            return []
            
        try:
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
        except Exception as e:
            st.error(f"Error splitting text into chunks: {str(e)}")
            return [text]  # Return original text as single chunk if splitting fails
    
    @st.cache_data
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

def main():
    st.set_page_config(page_title="Document QA without LLM", layout="wide")
    st.title("Document QA without LLM")
    st.write("Upload your documents and ask questions!")
    
    # Initialize the document processor
    @st.cache_resource
    def get_document_processor():
        return DocumentProcessor()
    
    doc_processor = get_document_processor()
    
    # File upload
    uploaded_files = st.file_uploader("Upload your documents (PDF/DOCX)", 
                                    type=["pdf", "docx"],
                                    accept_multiple_files=True)
    
    if uploaded_files:
        # Process documents
        progress_bar = st.progress(0)
        all_text = ""
        
        for i, file in enumerate(uploaded_files):
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            
            if file.type == "application/pdf":
                all_text += doc_processor.extract_text_from_pdf(file)
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                all_text += doc_processor.extract_text_from_docx(file)
                
        progress_bar.empty()
        
        if all_text.strip():
            # Split text into chunks
            chunks = doc_processor.split_into_chunks(all_text)
            
            if chunks:
                # Get embeddings for all chunks
                with st.spinner("Processing documents..."):
                    chunk_embeddings = doc_processor.get_embeddings(chunks)
                
                if len(chunk_embeddings) > 0:
                    st.success("Documents processed successfully!")
                    
                    # Query input
                    query = st.text_input("Ask a question about your documents:")
                    
                    if query:
                        with st.spinner("Searching for relevant information..."):
                            similar_chunks = doc_processor.find_most_similar_chunks(
                                query, chunks, chunk_embeddings
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

if __name__ == "__main__":
    main()
