import streamlit as st
from transformers import VisionEncoderDecoderModel, DonutProcessor
import torch
from PIL import Image, ImageDraw, ImageFont
import io

# ----------------------------
# Setup
# ----------------------------
st.title("DocVQA with Donut Model")

@st.cache_resource
def load_docvqa_model():
    model_name = "naver-clova-ix/donut-base-finetuned-docvqa"
    hf_token = "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"  # Directly using the provided token
    processor = DonutProcessor.from_pretrained(model_name, use_auth_token=hf_token)
    model = VisionEncoderDecoderModel.from_pretrained(model_name, use_auth_token=hf_token)
    return processor, model

processor, model = load_docvqa_model()

# ----------------------------
# Utility Functions
# ----------------------------
def chunk_text(text, chunk_size=200):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

def text_to_image(text):
    # Convert text chunk to an image for demonstration purposes.
    img = Image.new('RGB', (800, 1000), color='white')
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    margin = 10
    offset = 10
    for line in text.split('\n'):
        draw.text((margin, offset), line, font=font, fill='black')
        offset += 20
    return img

def run_docvqa_on_image(question, image: Image.Image):
    question_prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
    pixel_values = processor(image, return_tensors="pt").pixel_values
    decoder_input_ids = processor.tokenizer(question_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    outputs = model.generate(
        pixel_values, 
        decoder_input_ids=decoder_input_ids,
        max_length=512,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id
    )
    answer = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    answer = answer.replace("</s>", "").strip()
    if not answer:
        answer = "No clear answer found."
    return answer

# ----------------------------
# Streamlit App
# ----------------------------

st.write("Upload image documents (PNG/JPG) for best results. TXT files are supported but will not produce meaningful results since the model is trained on document images.")

uploaded_files = st.file_uploader("Upload documents (images or text)", 
                                  type=["txt","png","jpg","jpeg"],
                                  accept_multiple_files=True)

chunk_size = st.slider("Chunk size (for text documents)", min_value=50, max_value=500, value=200, step=50)

docs = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_type = uploaded_file.type
        if "text" in file_type:
            # Text file - chunk and convert chunks to images
            content = uploaded_file.read().decode("utf-8", errors='ignore')
            for ch in chunk_text(content, chunk_size):
                if ch.strip():
                    docs.append(('text', ch))
        elif "image" in file_type:
            # Image file - directly use
            image = Image.open(uploaded_file).convert("RGB")
            docs.append(('image', image))

if docs:
    user_query = st.text_input("Ask a question about the uploaded documents:")
    if user_query:
        # For simplicity, just use the first document
        doc_type, doc_data = docs[0]
        
        if doc_type == 'text':
            # Convert text chunk to image
            doc_image = text_to_image(doc_data)
            final_answer = run_docvqa_on_image(user_query, doc_image)
        else:
            # Already an image
            final_answer = run_docvqa_on_image(user_query, doc_data)

        st.subheader("Answer:")
        st.write(final_answer)

        # Show the document (or the rendered text image)
        if doc_type == 'text':
            st.subheader("Document Preview (Rendered from text):")
            st.image(text_to_image(doc_data))
        else:
            st.subheader("Document Preview:")
            st.image(doc_data)
else:
    st.write("No documents processed. Please upload at least one supported file.")
