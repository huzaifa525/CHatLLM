import streamlit as st
import requests
import json
import pytesseract
from PIL import Image
import io
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
import docx2txt
import PyPDF2
import re

# Set page configuration
st.set_page_config(page_title="Document Chat Assistant", layout="wide")

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'document_content' not in st.session_state:
    st.session_state.document_content = ""

def extract_text_from_image(image):
    """Extract text from image using Tesseract OCR"""
    try:
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"Error in OCR processing: {str(e)}")
        return ""

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error in PDF processing: {str(e)}")
        return ""

def extract_text_from_docx(docx_file):
    """Extract text from DOCX file"""
    try:
        text = docx2txt.process(docx_file)
        return text
    except Exception as e:
        st.error(f"Error in DOCX processing: {str(e)}")
        return ""

def format_numbers_with_latex(text):
    """Format numbers and mathematical expressions with LaTeX"""
    # Find numbers and mathematical expressions
    pattern = r'(\d+\.?\d*)|(\$[^$]+\$)'
    
    def replace_with_latex(match):
        if match.group(1):  # If it's a number
            return f"$${match.group(1)}$$"
        return match.group(0)  # If it's already LaTeX
    
    return re.sub(pattern, replace_with_latex, text)

def visualize_text_statistics(text):
    """Create visualizations for text statistics"""
    # Word count
    words = text.split()
    word_count = len(words)
    
    # Character frequency
    char_freq = pd.Series(list(text.lower())).value_counts()
    
    # Create visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Character Frequency Distribution")
        fig = px.bar(x=char_freq.index[:20], y=char_freq.values[:20])
        st.plotly_chart(fig)
    
    with col2:
        st.write("### Text Statistics")
        st.write(f"Total Words: $${word_count}$$")
        st.write(f"Total Characters: $${len(text)}$$")
        st.write(f"Average Word Length: $${np.mean([len(word) for word in words]):.2f}$$")

def get_ollama_response(prompt):
    """Get response from Ollama API"""
    try:
        response = requests.post(
            "http://147.182.201.56:11434/api/generate",
            json={
                "model": "smollm2:360m",
                "prompt": prompt,
                "stream": False
            }
        )
        return response.json()['response']
    except Exception as e:
        st.error(f"Error communicating with Ollama: {str(e)}")
        return "I apologize, but I'm having trouble connecting to the server."

# Sidebar for file upload
st.sidebar.title("Upload Document")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=['txt', 'pdf', 'docx', 'jpg', 'jpeg', 'png'])

# Process uploaded file
if uploaded_file is not None:
    file_type = uploaded_file.type
    
    if file_type.startswith('image'):
        image = Image.open(uploaded_file)
        text = extract_text_from_image(image)
    elif file_type == 'application/pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        text = extract_text_from_docx(uploaded_file)
    else:  # Assume text file
        text = uploaded_file.getvalue().decode()
    
    st.session_state.document_content = text
    
    # Show document content and visualizations
    st.write("### Document Content")
    st.write(format_numbers_with_latex(text))
    
    # Show visualizations
    visualize_text_statistics(text)

# Chat interface
st.write("### Chat")
if prompt := st.chat_input("Ask about the document"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Prepare context and get response
    context = f"Document content: {st.session_state.document_content}\nUser question: {prompt}"
    response = get_ollama_response(context)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(format_numbers_with_latex(message["content"]))
