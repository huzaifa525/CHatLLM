import streamlit as st
import PyPDF2
import docx
import torch
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Tuple, Dict
import re
from huggingface_hub import HfApi

class DocumentQA:
    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        # Initialize models
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', use_auth_token=hf_token)
        self.encoder = GPT2Model.from_pretrained('gpt2', use_auth_token=hf_token)
        self.generator = GPT2LMHeadModel.from_pretrained('gpt2', use_auth_token=hf_token)
        
        # Set models to evaluation mode
        self.encoder.eval()
        self.generator.eval()
        
        # Initialize storage
        self.chunks = []
        self.embeddings = None
        self.bm25 = None
        
        # Add special tokens
        special_tokens = {
            'pad_token': '<|pad|>',
            'sep_token': '<|sep|>',
            'question_token': '<|question|>'
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.encoder.resize_token_embeddings(len(self.tokenizer))
        self.generator.resize_token_embeddings(len(self.tokenizer))

    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text while preserving structure"""
        # Remove extra whitespace while preserving paragraph breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def create_semantic_chunks(self, text: str, max_length: int = 512) -> List[str]:
        """Create semantically meaningful chunks of text"""
        # Split into paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            # Get token count for current paragraph
            tokens = self.tokenizer.encode(para)
            para_length = len(tokens)
            
            if current_length + para_length > max_length:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [para]
                current_length = para_length
            else:
                current_chunk.append(para)
                current_length += para_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def get_contextual_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Generate contextual embeddings using GPT-2"""
        all_embeddings = []
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors='pt',
                                      truncation=True, max_length=512,
                                      padding=True)
                outputs = self.encoder(**inputs)
                # Use mean pooling of last hidden states
                embeddings = outputs.last_hidden_state.mean(dim=1)
                all_embeddings.append(embeddings)
                
        return torch.cat(all_embeddings, dim=0)

    def initialize_qa(self, text: str):
        """Initialize the QA system with document text"""
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Create semantic chunks
        self.chunks = self.create_semantic_chunks(processed_text)
        
        # Generate embeddings
        self.embeddings = self.get_contextual_embeddings(self.chunks)
        
        # Initialize BM25
        tokenized_chunks = [chunk.split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer using GPT-2"""
        # Prepare input text
        input_text = f"{context}{self.tokenizer.question_token}{question}"
        
        # Tokenize
        inputs = self.tokenizer(input_text, return_tensors='pt',
                              truncation=True, max_length=512)
        
        # Generate answer
        with torch.no_grad():
            outputs = self.generator.generate(
                inputs.input_ids,
                max_length=150,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode and clean answer
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = answer[len(input_text):].strip()
        return answer

    def answer_question(self, question: str, top_k: int = 3) -> List[Dict]:
        """Answer question using retrieved contexts"""
        if not self.chunks:
            return []
        
        # Get question embedding
        question_embedding = self.get_contextual_embeddings([question])
        
        # Calculate similarity scores
        similarities = torch.nn.functional.cosine_similarity(
            question_embedding,
            self.embeddings
        )
        
        # Get BM25 scores
        bm25_scores = self.bm25.get_scores(question.split())
        bm25_scores = torch.tensor(bm25_scores)
        
        # Normalize and combine scores
        similarities = (similarities - similarities.min()) / (similarities.max() - similarities.min())
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
        combined_scores = 0.7 * similarities + 0.3 * bm25_scores
        
        # Get top chunks
        top_indices = torch.topk(combined_scores, min(top_k, len(self.chunks))).indices
        
        results = []
        for idx in top_indices:
            context = self.chunks[idx]
            score = combined_scores[idx].item()
            answer = self.generate_answer(question, context)
            
            results.append({
                'answer': answer,
                'context': context,
                'confidence': score
            })
            
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
    st.set_page_config(page_title="GPT-2 Document QA", layout="wide")
    st.title("Document Question Answering System")
    
    # HuggingFace token input
    hf_token = st.text_input("Enter your HuggingFace token:", type="password")
    
    if not hf_token:
        st.warning("Please enter your HuggingFace token to proceed.")
        st.info("Get your token from: https://huggingface.co/settings/tokens")
        return
        
    if not validate_token(hf_token):
        st.error("Invalid HuggingFace token. Please check and try again.")
        return
    
    # Initialize QA system
    if 'qa_system' not in st.session_state:
        with st.spinner("Loading GPT-2 models..."):
            st.session_state.qa_system = DocumentQA(hf_token)
    
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
                        text += page.extract_text() + "\n\n"
                elif file.name.endswith(('.docx', '.doc')):
                    doc = docx.Document(file)
                    for para in doc.paragraphs:
                        text += para.text + "\n\n"
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
        
        if text.strip():
            with st.spinner("Processing documents..."):
                st.session_state.qa_system.initialize_qa(text)
                st.success("Documents processed successfully!")
        
        # Question input
        question = st.text_input("Ask a question about your documents:")
        
        if question:
            with st.spinner("Generating answer..."):
                results = st.session_state.qa_system.answer_question(question)
            
            if results:
                st.subheader("Answers:")
                for i, result in enumerate(results, 1):
                    with st.container():
                        st.markdown("---")
                        st.markdown(f"**Answer {i}:**")
                        st.write(result['answer'])
                        st.markdown(f"**Confidence Score:** {result['confidence']:.4f}")
                        with st.expander("Show Context"):
                            st.write(result['context'])
            else:
                st.warning("Could not generate an answer. Please try rephrasing your question.")
    
    # Add clear button
    if st.button("Clear All"):
        st.session_state.clear()
        st.experimental_rerun()

if __name__ == "__main__":
    main()
