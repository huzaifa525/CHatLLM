import streamlit as st
from transformers import VisionEncoderDecoderModel, DonutProcessor
import torch
from PIL import Image
import io

# ----------------------------
# Setup
# ----------------------------
st.title("DocVQA with Donut Model")

@st.cache_resource
def load_docvqa_model():
    # Load Donut model and processor
    model_name = "naver-clova-ix/donut-base-finetuned-docvqa"
    processor = DonutProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    return processor, model

processor, model = load_docvqa_model()

# ----------------------------
# Utility Functions
# ----------------------------
def chunk_text(text, chunk_size=200):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

def run_docvqa_on_image(question, image: Image.Image):
    # Prepare question prompt as per Donut docvqa format
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
    return answer if answer else "No clear answer found."

def text_to_image(text):
    # Convert text chunk to an image for demonstration
    # (This is a hack. The donut model is for images, not raw text.)
    # For demonstration, we will render text on an image background.
    # Results may not be meaningful since the model is trained on document images.
    
    from PIL import ImageDraw, ImageFont
    # Create a white image
    img = Image.new('RGB', (800, 1000), color='white')
    draw = ImageDraw.Draw(img)
    # If you have a font file, you can specify it, else will use default.
    # Attempt to wrap text
    font = ImageFont.load_default()
    margin = 10
    offset = 10
    for line in text.split('\n'):
        draw.text((margin, offset), line, font=font, fill='black')
        offset += 20
    return img

# ----------------------------
# Streamlit App
# ----------------------------

st.write("Upload image or text documents and then ask a question. For best results, upload scanned document images.")

uploaded_files = st.file_uploader("Upload documents (images like .png/.jpg or text files)", 
                                  type=["txt","png","jpg","jpeg"],
                                  accept_multiple_files=True)

chunk_size = st.slider("Chunk size (for text documents)", min_value=50, max_value=500, value=200, step=50)

docs = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_type = uploaded_file.type
        if "text" in file_type:
            # Text file, chunk it and convert chunks to images
            content = uploaded_file.read().decode("utf-8", errors='ignore')
            for ch in chunk_text(content, chunk_size):
                if ch.strip():
                    docs.append(('text', ch))
        elif "image" in file_type:
            # Direct image
            image = Image.open(uploaded_file).convert("RGB")
            docs.append(('image', image))

if docs:
    user_query = st.text_input("Ask a question about the uploaded documents:")
    if user_query:
        # For simplicity, just use the first doc. 
        # If multiple docs are present, you could let the user pick which doc to query.
        doc_type, doc_data = docs[0]
        
        if doc_type == 'text':
            # Convert text chunk to image
            doc_image = text_to_image(doc_data)
            final_answer = run_docvqa_on_image(user_query, doc_image)
        else:
            # It's already an image
            final_answer = run_docvqa_on_image(user_query, doc_data)

        st.subheader("Answer:")
        st.write(final_answer)

        # Optionally show the doc image
        if doc_type == 'text':
            st.subheader("Document Preview (Rendered from text):")
            st.image(text_to_image(doc_data))
        else:
            st.subheader("Document Preview:")
            st.image(doc_data)
else:
    st.write("No documents processed. Please upload an image or text file.")
