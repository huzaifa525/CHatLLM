import streamlit as st
import requests
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from io import BytesIO
import textract
import re
import os
import tempfile

# Set Tesseract executable path (update to match your installation)
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Define Ollama API endpoint and model
OLLAMA_URL = "http://147.182.201.56:11434/api"
MODEL_NAME = "smollm2:360m"

def query_ollama(message):
    """Send a query to the Ollama model."""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": message}
        ]
    }
    response = requests.post(f"{OLLAMA_URL}/chat", json=payload)
    if response.status_code == 200:
        return response.json().get("response", "No response")
    else:
        return f"Error: {response.status_code} - {response.text}"

def extract_text_from_file(uploaded_file):
    """Extract text from uploaded files."""
    if uploaded_file.type == "application/pdf":
        pages = convert_from_path(uploaded_file)
        text = "\n".join(pytesseract.image_to_string(page) for page in pages)
    elif uploaded_file.type.startswith("image"):
        image = Image.open(uploaded_file)
        text = pytesseract.image_to_string(image)
    elif uploaded_file.type in ("text/plain", "application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        try:
            text = textract.process(temp_file_path).decode("utf-8")
        finally:
            os.remove(temp_file_path)  # Ensure the temporary file is deleted
    else:
        text = "Unsupported file type."
    return text

def visualize_text_with_latex(text):
    """Format text with LaTeX if applicable."""
    text = re.sub(r'\$([^$]+)\$', r'$$\1$$', text)  # Handle inline LaTeX
    return text

# Streamlit UI
def main():
    st.set_page_config(page_title="Ollama Chatbot", layout="wide")
    st.title("Chatbot with Document Upload")

    # File upload
    st.sidebar.header("Upload Document")
    uploaded_file = st.sidebar.file_uploader("Upload DOC, TXT, JPG, or PDF", type=["doc", "docx", "txt", "jpg", "jpeg", "png", "pdf"])

    # Placeholder for chat
    chat_placeholder = st.container()

    # Document text placeholder
    document_text = ""

    if uploaded_file:
        with st.spinner("Extracting text from the uploaded document..."):
            document_text = extract_text_from_file(uploaded_file)
        st.sidebar.success("Document uploaded and text extracted!")

        # Show extracted text in the sidebar
        with st.sidebar.expander("Extracted Text"):
            st.text_area("Text from Document", document_text, height=200)

    # Chat functionality
    with chat_placeholder:
        st.header("Chat with your Document")

        if document_text:
            st.markdown("Start chatting with the content extracted from your document.")
        else:
            st.markdown("Upload a document to begin.")

        # User input
        user_input = st.text_input("Your Message:")
        if st.button("Send") and user_input:
            with st.spinner("Sending query to Ollama..."):
                response = query_ollama(user_input + "\n" + document_text)
                formatted_response = visualize_text_with_latex(response)

                st.markdown(f"### **Chatbot Response:**")
                st.write(formatted_response)

if __name__ == "__main__":
    main()
