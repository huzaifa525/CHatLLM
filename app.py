import streamlit as st
import PyPDF2
import docx
import torch
from transformers import GPT2Tokenizer, GPT2Model
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Tuple
import re
from huggingface_hub import HfApi
import time

class GPT2Retriever:
    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        # Initialize GPT2 model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', use_auth_token=hf_token)
        self.model = GPT2Model.from_pretrained('gpt2', use_auth_token=hf_token)
        self.model.eval()  # Set to evaluation mode
        self.chunks = []
        self.embeddings = None
        self.bm25 = None

    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Get GPT-2 embeddings for texts"""
        all_embeddings = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize and get model output
                inputs = self.tokenizer(text, return_tensors='pt', 
                                     truncation=True, max_length=512,
                                     padding=True)
                outputs = self.model(**inputs)
                
                # Use the last hidden state's mean as embedding
                last_hidden_state = outputs.last_hidden_state
                mean_embedding = torch.mean(last_hidden_state, dim=1)
                all_embeddings.append(mean_embedding)
        
        return torch.cat(all_embeddings, dim=0)

    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing"""
        # Preserve special patterns (invoice numbers, dates, etc.)
        text = re.sub(r'(\b\w+[-/]\w+\b|\b\d+[A-Z]\d+[A-Z]\d+\b|\b\d{6,}\b)', r' \1 ', text)
        
        # Remove excessive whitespace while preserving important separators
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text

    def create_chunks(self, text: str, max_length: int = 512) -> List[str]:
        """Create overlapping chunks considering token limits"""
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            # Get token count for current sentence
            tokens = len(self.tokenizer.encode(sentence))
            
            if current_length + tokens > max_length:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = tokens
            else:
                current_chunk.append(sentence)
                current_length += tokens
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def extract_patterns(self, text: str) -> List[str]:
        """Extract potential invoice numbers and important patterns"""
        patterns = [
            r'\b\d{6,}\b',  # 6 or more digits
            r'\b[A-Z0-9]{8,}\b',  # Alphanumeric 8 or more chars
            r'\b\w+[-/]\w+\b',  # Pattern with dash or slash
            r'\b\d+[A-Z]\d+[A-Z]\d+\b',  # Mixed digit-letter pattern
            r'\b[A-Z0-9]{8}(?:-[A-Z0-9]{4}){3}-[A-Z0-9]{12}\b'  # UUID-like pattern
        ]
        
        found_patterns = []
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                found_patterns.append(match.group())
        
        return found_patterns

    def initialize_retriever(self, text: str):
        """Initialize the retriever with processed text"""
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Create chunks
        self.chunks = self.create_chunks(processed_text)
        
        # Initialize BM25
        tokenized_chunks = [chunk.split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
        
        # Generate embeddings
        self.embeddings = self.get_embeddings(self.chunks)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float, List[str]]]:
        """Enhanced search with pattern extraction"""
        if not self.chunks:
            return []

        # Get query embedding
        query_embedding = self.get_embeddings([query])
        
        # Calculate similarity scores
        similarities = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0), 
            self.embeddings
        )
        
        # Get BM25 scores
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_scores = torch.tensor(bm25_scores)
        
        # Normalize scores
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
        similarities = (similarities - similarities.min()) / (similarities.max() - similarities.min())
        
        # Combine scores
        combined_scores = 0.6 * similarities + 0.4 * bm25_scores
        
        # Get top results
        top_k = min(top_k, len(self.chunks))
        top_indices = torch.topk(combined_scores, top_k).indices
        
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            patterns = self.extract_patterns(chunk)
            score = combined_scores[idx].item()
            results.append((chunk, score, patterns))
        
        return results

def validate_token(token: str) -> bool:
    """Validate HuggingFace token"""
    try:
        api = HfApi(token=token)
        api.whoami()
        return True
    except Exception:
        return False

def main():
    st.set_page_config(page_title="GPT-2 Enhanced Document QA", layout="wide")
    st.title("GPT-2 Enhanced Document QA")
    
    # HuggingFace token input
    hf_token = st.text_input("Enter your HuggingFace token:", type="password")
    
    if not hf_token:
        st.warning("Please enter your HuggingFace token to proceed.")
        st.info("Get your token from: https://huggingface.co/settings/tokens")
        return
        
    if not validate_token(hf_token):
        st.error("Invalid HuggingFace token. Please check and try again.")
        return
    
    # Initialize retriever in session state
    if 'retriever' not in st.session_state:
        with st.spinner("Loading GPT-2 model..."):
            st.session_state.retriever = GPT2Retriever(hf_token)
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload documents (PDF/DOCX)", 
        type=["pdf", "docx"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Process documents
        text = ""
        for file in uploaded_files:
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
        
        if text.strip():
            with st.spinner("Processing documents with GPT-2..."):
                st.session_state.retriever.initialize_retriever(text)
                st.success("Documents processed successfully!")
        
        # Query interface
        query = st.text_input("Ask a question about your documents:")
        
        if query:
            with st.spinner("Searching with GPT-2..."):
                results = st.session_state.retriever.search(query)
            
            if results:
                st.subheader("Found Information:")
                for chunk, score, patterns in results:
                    with st.container():
                        st.markdown("---")
                        st.markdown(f"**Confidence Score:** {score:.4f}")
                        if patterns:
                            st.markdown("**Important Numbers/IDs Found:**")
                            for pattern in patterns:
                                st.markdown(f"- `{pattern}`")
                        st.markdown("**Context:**")
                        st.write(chunk)
            else:
                st.warning("No relevant information found.")
    
    # Add clear button
    if st.button("Clear All"):
        st.session_state.clear()
        st.experimental_rerun()

if __name__ == "__main__":
    main()
