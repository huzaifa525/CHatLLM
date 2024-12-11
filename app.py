import streamlit as st
from transformers import VisionEncoderDecoderModel, DonutProcessor
import torch
from PIL import Image, ImageDraw, ImageFont
import io
import os
from typing import Tuple, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Environment and Config
# ----------------------------
def get_hf_token() -> str:
    """Securely get HuggingFace token from Streamlit secrets."""
    try:
        return st.secrets["hf_token"]
    except Exception as e:
        st.error("""
        HuggingFace token not found in Streamlit secrets.
        Please add your token to the secrets.toml file in your Streamlit Cloud deployment.
        """)
        st.stop()

# ----------------------------
# Model Loading
# ----------------------------
@st.cache_resource
def load_docvqa_model() -> Tuple[DonutProcessor, VisionEncoderDecoderModel]:
    """
    Load the DocVQA model with error handling and caching.
    Returns:
        Tuple[DonutProcessor, VisionEncoderDecoderModel]: Processor and model
    """
    try:
        model_name = "naver-clova-ix/donut-base-finetuned-docvqa"
        hf_token = get_hf_token()
        
        logger.info("Loading DonutProcessor...")
        processor = DonutProcessor.from_pretrained(
            model_name,
            use_auth_token=hf_token,
            trust_remote_code=True
        )
        
        logger.info("Loading VisionEncoderDecoderModel...")
        model = VisionEncoderDecoderModel.from_pretrained(
            model_name,
            use_auth_token=hf_token,
            trust_remote_code=True
        )
        
        if torch.cuda.is_available():
            model = model.to("cuda")
            logger.info("Model moved to GPU")
        
        return processor, model
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error("""
        Error loading the DocVQA model. Please ensure:
        1. You have a valid HuggingFace token set in environment variables
        2. You have sufficient internet connectivity
        3. The model is accessible with your credentials
        """)
        st.stop()

# ----------------------------
# Utility Functions
# ----------------------------
def chunk_text(text: str, chunk_size: int = 200) -> List[str]:
    """Split text into chunks of specified size."""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def text_to_image(text: str, width: int = 800, height: int = 1000) -> Image.Image:
    """Convert text to an image with improved formatting."""
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    try:
        # Try to load a better font if available
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    margin = 20
    offset = 20
    max_width = width - 2 * margin
    
    for line in text.split('\n'):
        # Basic word wrapping
        words = line.split()
        current_line = []
        current_width = 0
        
        for word in words:
            word_width = draw.textlength(word + " ", font=font)
            if current_width + word_width <= max_width:
                current_line.append(word)
                current_width += word_width
            else:
                draw.text((margin, offset), " ".join(current_line), font=font, fill='black')
                offset += 20
                current_line = [word]
                current_width = word_width
        
        if current_line:
            draw.text((margin, offset), " ".join(current_line), font=font, fill='black')
            offset += 25
    
    return img

def run_docvqa_on_image(
    question: str,
    image: Image.Image,
    processor: DonutProcessor,
    model: VisionEncoderDecoderModel
) -> str:
    """
    Run DocVQA inference with improved error handling and validation.
    """
    try:
        # Validate inputs
        if not isinstance(image, Image.Image):
            raise ValueError("Invalid image input")
        if not question.strip():
            raise ValueError("Empty question")
            
        # Prepare inputs
        question_prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
        pixel_values = processor(image, return_tensors="pt", legacy=False).pixel_values
        
        if torch.cuda.is_available():
            pixel_values = pixel_values.to("cuda")
            
        decoder_input_ids = processor.tokenizer(
            question_prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids
        
        if torch.cuda.is_available():
            decoder_input_ids = decoder_input_ids.to("cuda")

        # Generate answer
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=512,
            num_beams=4,
            early_stopping=True,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id
        )

        # Process output
        answer = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        answer = answer.replace("</s>", "").strip()
        
        return answer if answer else "No clear answer found."
        
    except Exception as e:
        logger.error(f"Error in DocVQA inference: {str(e)}")
        return f"Error processing question: {str(e)}"

# ----------------------------
# Streamlit App
# ----------------------------
def main():
    st.title("📄 DocVQA with Donut Model")
    st.write("""
    Upload document images (PNG/JPG) to ask questions about their content.
    For text files, the content will be rendered as images for processing.
    """)

    # Load models
    with st.spinner("Loading DocVQA model..."):
        processor, model = load_docvqa_model()

    # File upload
    uploaded_files = st.file_uploader(
        "Upload documents (images or text)",
        type=["txt", "png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Please upload at least one document to begin.")
        return

    # Process uploads
    docs = []
    with st.spinner("Processing uploaded documents..."):
        for uploaded_file in uploaded_files:
            try:
                if uploaded_file.type == "text/plain":
                    content = uploaded_file.read().decode("utf-8", errors='ignore')
                    chunk_size = st.slider(
                        "Text chunk size",
                        min_value=50,
                        max_value=500,
                        value=200,
                        step=50
                    )
                    for chunk in chunk_text(content, chunk_size):
                        if chunk.strip():
                            docs.append(('text', chunk))
                else:
                    image = Image.open(uploaded_file).convert("RGB")
                    docs.append(('image', image))
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")

    if not docs:
        st.error("No valid documents were processed.")
        return

    # Question answering
    user_query = st.text_input("💭 Ask a question about the uploaded documents:")
    
    if user_query:
        with st.spinner("Analyzing document..."):
            # For simplicity, use the first document
            doc_type, doc_data = docs[0]
            
            if doc_type == 'text':
                doc_image = text_to_image(doc_data)
            else:
                doc_image = doc_data

            answer = run_docvqa_on_image(user_query, doc_image, processor, model)
            
            st.subheader("📝 Answer:")
            st.write(answer)

            # Show document preview
            st.subheader("📄 Document Preview:")
            st.image(doc_image, use_column_width=True)

if __name__ == "__main__":
    main()
