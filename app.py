import streamlit as st
import PyPDF2
import docx
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import torch
from typing import List, Tuple, Dict
import re
from transformers import AutoTokenizer
from huggingface_hub import HfApi
import time
from tqdm import tqdm

class EnhancedRetriever:
    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        # Using a more powerful SBERT model for better semantic understanding
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', 
                                        use_auth_token=hf_token)
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2',
                                                      use_auth_token=hf_token)
        self.bm25 = None
        self.chunks = []
        self.embeddings = None
        
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text while preserving important invoice/receipt numbers"""
        # Preserve receipt/invoice numbers with specific patterns
        text = re.sub(r'(\b\w+[-/]\w+\b|\b\d+[A-Z]\d+[A-Z]\d+\b|\b\d{6,}\b)', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)  # Normalize spaces
        return text.strip()
        
    def extract_invoice_number(self, text: str) -> List[str]:
        """Extract potential invoice/receipt numbers"""
        patterns = [
            r'\b\d{6,}\b',  # 6 or more digits
            r'\b[A-Z0-9]{8,}\b',  # Alphanumeric 8 or more chars
            r'\b\w+[-/]\w+\b',  # Pattern with dash or slash
            r'\b\d+[A-Z]\d+[A-Z]\d+\b'  # Mixed digit-letter pattern
        ]
        numbers = []
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                numbers.append(match.group())
    
    def create_sliding_windows(self, text: str, window_size: int = 100, stride: int = 50) -> List[str]:
        """Create overlapping windows of text for better context preservation"""
        words = text.split()
        windows = []
        
        for i in range(0, len(words), stride):
            window = ' '.join(words[i:i + window_size])
            if window:
                windows.append(window)
                
        return windows
    
    def chunk_text(self, text: str) -> List[str]:
        """Enhanced text chunking with sliding windows and smart splitting"""
        # First clean the text
        text = self.preprocess_text(text)
        
        # Split into paragraphs
        paragraphs = text.split('\n')
        
        # Process each paragraph
        chunks = []
        for para in paragraphs:
            if len(para.split()) > 100:  # Long paragraph
                chunks.extend(self.create_sliding_windows(para))
            else:
                chunks.append(para)
                
        # Remove duplicates while preserving order
        seen = set()
        filtered_chunks = []
        for chunk in chunks:
            if chunk not in seen and len(chunk.split()) > 5:  # Minimum chunk size
                seen.add(chunk)
                filtered_chunks.append(chunk)
                
        return filtered_chunks
    
    def initialize_retriever(self, text: str):
        """Initialize both dense and sparse retrievers"""
        self.chunks = self.chunk_text(text)
        
        # Initialize BM25
        tokenized_chunks = [chunk.split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
        
        # Generate dense embeddings
        self.embeddings = self.model.encode(self.chunks, 
                                          show_progress_bar=True, 
                                          batch_size=8,
                                          normalize_embeddings=True)  # Normalized for better similarity
    
    def hybrid_search(self, query: str, top_k: int = 3) -> List[Tuple[str, float, List[str]]]:
        """Combine BM25 and dense retrieval for more accurate results"""
        if not self.chunks:
            return []
            
        # Get BM25 scores
        bm25_scores = self.bm25.get_scores(query.split())
        if len(bm25_scores) > 0:
            bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))
        
        # Get dense embedding scores
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
        dense_scores = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Adjust weights based on query type
        if any(keyword in query.lower() for keyword in ['invoice', 'receipt', 'number', 'id']):
            combined_scores = 0.6 * bm25_scores + 0.4 * dense_scores
        else:
            combined_scores = 0.3 * bm25_scores + 0.7 * dense_scores
            
        # Extract invoice numbers from relevant chunks
        results = []
        top_indices = combined_scores.argsort()[-top_k:][::-1]
        
        for idx in top_indices:
            chunk = self.chunks[idx]
            invoice_numbers = self.extract_invoice_number(chunk)
            results.append((chunk, combined_scores[idx], invoice_numbers))
        
        # Get top results
        top_indices = combined_scores.argsort()[-top_k:][::-1]
        results = [(self.chunks[i], combined_scores[i]) for i in top_indices]
        
        return results

def extract_text_from_docs(files) -> str:
    """Extract text from multiple document formats"""
    text = ""
    for file in files:
        try:
            if file.name.endswith('.pdf'):
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            elif file.name.endswith(('.docx', '.doc')):
                doc = docx.Document(file)
                for para in doc.paragraphs:
                    text += para.text + "\n"
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
    return text

def validate_token(token: str) -> bool:
    """Validate HuggingFace token"""
    try:
        api = HfApi(token=token)
        api.whoami()
        return True
    except Exception:
        return False

def main():
    st.set_page_config(page_title="High-Accuracy Document QA", layout="wide")
    st.title("High-Accuracy Document QA System")
    
    # HuggingFace token input
    hf_token = st.text_input("Enter your HuggingFace token:", type="password")
    
    if not hf_token:
        st.warning("Please enter your HuggingFace token to proceed.")
        st.info("You can get your token from: https://huggingface.co/settings/tokens")
        return
        
    if not validate_token(hf_token):
        st.error("Invalid HuggingFace token. Please check and try again.")
        return
    
    # Initialize retriever in session state
    if 'retriever' not in st.session_state:
        st.session_state.retriever = EnhancedRetriever(hf_token)
    
    # File upload
    uploaded_files = st.file_uploader("Upload documents (PDF/DOCX)", 
                                    type=["pdf", "docx"], 
                                    accept_multiple_files=True)
    
    if uploaded_files:
        with st.spinner("Processing documents..."):
            text = extract_text_from_docs(uploaded_files)
            
            if text.strip():
                st.session_state.retriever.initialize_retriever(text)
                st.success("Documents processed successfully!")
            else:
                st.error("No text could be extracted from the documents.")
                return
    
        # Query interface
        query = st.text_input("Ask a question about your documents:")
        
        if query:
            with st.spinner("Searching for relevant information..."):
                results = st.session_state.retriever.hybrid_search(query, top_k=3)
                
            if results:
                st.subheader("Invoice/Receipt Numbers Found:")
                for chunk, score, invoice_numbers in results:
                    if invoice_numbers:
                        with st.container():
                            st.markdown("---")
                            st.markdown(f"**Confidence Score:** {score:.4f}")
                            st.markdown("**Found Numbers:**")
                            for num in invoice_numbers:
                                st.markdown(f"- {num}")
                            st.markdown("**Context:**")
                            st.write(chunk)
                        
                # Provide similarity threshold warning
                if all(score < 0.5 for _, score in results):
                    st.warning("⚠️ The confidence scores are relatively low. The answers might not be fully relevant.")
            else:
                st.warning("No relevant information found in the documents.")
    
    # Add clear button
    if st.button("Clear All"):
        st.session_state.clear()
        st.experimental_rerun()

if __name__ == "__main__":
    main()
