import streamlit as st
import nltk
import os
from pathlib import Path
from nltk.corpus import stopwords
import re
from googletrans import Translator
from textblob import TextBlob
import requests
import langdetect
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from camel_tools.utils.normalize import normalize_unicode
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os


try:
    nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)
    
for resource in ['stopwords', 'punkt']:
   try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception as e:
    st.warning(f"Failed to setup NLTK directory. Some features might be limited. Error: {str(e)}")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Add logging settings
from transformers import logging
logging.set_verbosity_error()

translator = Translator()

@st.cache_resource
def load_summarization_models():
    return {
        'en': {
            'small': pipeline("summarization", model="facebook/bart-large-cnn"),
            'base': pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
        },
        'ar': {
            'small': pipeline("summarization", model="csebuetnlp/mT5_multilingual_XLSum"),
            'base': (
                AutoTokenizer.from_pretrained("UBC-NLP/AraBART", use_auth_token=True),
                AutoModelForSeq2SeqLM.from_pretrained("UBC-NLP/AraBART", use_auth_token=True)
            )
        }
    }

summarization_models = load_summarization_models()

def arabic_sentence_tokenize(text):
    endings = ['؟', '!', '.', '؛', '\n', '\r\n']
    pattern = '|'.join(map(re.escape, endings))
    sentences = [s.strip() for s in re.split(f'(?<=[{pattern}])', text) if s.strip()]
    return sentences

def arabic_word_tokenize(text):
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    words = re.findall(r'[\u0600-\u06FF]+|[a-zA-Z]+', text)
    return words

def preprocess_text(text, language='ar'):
    if language == 'ar':
        text = re.sub(r'[^\u0600-\u06FF\s\.\!\؟\،\؛]', '', text)
    else:
        text = re.sub(r'[^a-zA-Z0-9\s\.\!\?]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def transformer_summarize(text, language='ar', model_size='base', max_length=300):
    try:
        if language == 'en':
            if model_size == 'small':
                summary = summarization_models['en']['small'](
                    text, 
                    max_length=max_length, 
                    min_length=30, 
                    do_sample=False
                )[0]['summary_text']
            else:
                summary = summarization_models['en']['base'](
                    text, 
                    max_length=max_length, 
                    min_length=30, 
                    do_sample=False
                )[0]['summary_text']
        elif language == 'ar':
            if model_size == 'small':
                summary = summarization_models['ar']['small'](
                    text, 
                    max_length=max_length, 
                    min_length=30, 
                    do_sample=False
                )[0]['summary_text']
            else:
                tokenizer, model = summarization_models['ar']['base']
                inputs = tokenizer(
                    text, 
                    return_tensors="pt", 
                    max_length=1024, 
                    truncation=True
                )
                summary_ids = model.generate(
                    inputs["input_ids"],
                    num_beams=4,
                    max_length=max_length,
                    early_stopping=True
                )
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        st.error(f"Transformer summarization failed: {str(e)}")
        return None

def summarize_text(text, num_sentences=3, language='ar', use_transformer=True):
    if not text:
        return ""
    
    try:
        text = preprocess_text(text, language)
        
        if use_transformer:
            with st.spinner('Generating advanced summary...'):
                max_length = min(len(text)//4, 1024)
                summary = transformer_summarize(
                    text, 
                    language=language,
                    max_length=max_length
                )
                if summary:
                    return summary
        
        # Fallback to extractive method
        if language == 'ar':
            sentences = arabic_sentence_tokenize(text)
            words = arabic_word_tokenize(text)
            stop_words = set(stopwords.words('arabic')) if hasattr(stopwords, 'words') else set()
        else:
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            stop_words = set(stopwords.words('english'))
        
        words = [word for word in words if word.lower() not in stop_words]
        freq_table = Counter(words)
        
        sentence_scores = {}
        for sentence in sentences:
            for word, freq in freq_table.items():
                if word in sentence:
                    sentence_scores[sentence] = sentence_scores.get(sentence, 0) + freq
        
        num_sentences = min(num_sentences, len(sentences))
        if sentence_scores:
            summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
            return ' '.join(summary_sentences)
        else:
            return ' '.join(sentences[:num_sentences])
    
    except Exception as e:
        st.error(f"Summarization error: {str(e)}")
        return text[:500] + "..."

def analyze_sentiment(text):
    """
    Analyze sentiment of the text
    """
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment.polarity, sentiment.subjectivity

def extract_keywords(text, num_keywords=5):
    """
    Extract keywords from the text
    """
    if not text:
        return []
    
    # Preprocess the text
    text = preprocess_text(text, language='ar' if langdetect.detect(text) == 'ar' else 'en')
    
    # Tokenize words
    words = arabic_word_tokenize(text) if langdetect.detect(text) == 'ar' else word_tokenize(text)
    
    # Remove stopwords
    try:
        stop_words = set(stopwords.words('arabic')) if langdetect.detect(text) == 'ar' else set(stopwords.words('english'))
    except:
        stop_words = set()
    
    words = [word for word in words if word.lower() not in stop_words and len(word) > 3]
    
    # Count word frequencies
    word_freq = Counter(words)
    
    # Get the top N keywords
    return [word for word, _ in word_freq.most_common(num_keywords)]

def identify_hard_words(text, language='ar'):
    if language == 'ar':
        detected_language = langdetect.detect(text)
        if detected_language != 'ar':
            text = translate_text(text, src=detected_language, dest='ar')
        words = arabic_word_tokenize(text)
        try:
            stop_words = set(stopwords.words('arabic'))
        except:
            stop_words = set()
    else:
        words = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
    
    hard_words = [word for word in words if word.lower() not in stop_words and len(word) > 3]
    unique_hard_words = list(set(hard_words))
    
    hard_word_info = {}
    for word in unique_hard_words:
        translated_word = translate_text(word, src='ar' if language == 'ar' else 'en', 
                                      dest='en' if language == 'ar' else 'ar')
        example_sentence = fetch_example_sentence(word, language)
        hard_word_info[word] = {
            'translation': translated_word,
            'example': example_sentence
        }
    return hard_word_info

def fetch_example_sentence(term, language='ar'):
    """
    Fetch an example sentence from the internet for a given term
    """
    if language == 'ar':
        term_en = translator.translate(term, src='ar', dest='en').text
    else:
        term_en = term
    
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{term_en}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data:
                meanings = data[0].get('meanings', [])
                for meaning in meanings:
                    examples = meaning.get('definitions', [{}])[0].get('example', None)
                    if examples:
                        return examples
    except Exception as e:
        st.error(f"Error fetching example sentence: {str(e)}")
    return f"No example found for {term}"

def translate_text(text, src='ar', dest='en'):
    """
    Translate text using Google Translate
    """
    try:
        translated = translator.translate(text, src=src, dest=dest)
        return translated.text
    except Exception as e:
        st.error(f"Error translating text: {str(e)}")
        return text

def fetch_website_content(url):
    """
    Fetch content from a website
    """
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Improve content extraction by considering additional tags
            content_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']
            content_elements = []
            for tag in content_tags:
                content_elements.extend(soup.find_all(tag))
            
            text = ' '.join([element.get_text(strip=True) for element in content_elements])
            return text
        else:
            st.error(f"Failed to fetch content from {url}")
    except Exception as e:
        st.error(f"Error fetching website content: {str(e)}")
    return ""

def read_pdf(file_stream):
    """
    Read text from a PDF file
    """
    try:
        pdf_document = fitz.open(stream=file_stream, filetype="pdf")
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text("text")
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def ocr_image(image_stream):
    """
    Perform OCR on an image stream
    """
    try:
        image = Image.open(image_stream)
        text = pytesseract.image_to_string(image, lang='eng')
        return text
    except Exception as e:
        st.error(f"Error performing OCR: {str(e)}")
        return ""

# Streamlit UI
st.title("Sarsor 🤖")

# Sidebar controls
with st.sidebar:
    st.header("Settings ⚙️")
    summary_length = st.slider("Summary Length", 1, 10, 3)
    model_size = st.radio("Model Size", ['base', 'small'], help="Base models are more accurate but slower")
    use_transformer = st.checkbox("Use Deep Learning Models", value=True)
    auto_translate = st.checkbox("Auto-translate Summary")

# Ensure Arabic text is displayed in RTL format
st.markdown("""
<style>
div[data-testid="stMarkdownContainer"] p {
    direction: rtl;
}
</style>
""", unsafe_allow_html=True)

# Input Method
input_option = st.selectbox("Input Method", ["Upload File (PDF/Image)", "Paste Text", "Enter Website URL"])

if input_option == "Upload File (PDF/Image)":
    uploaded_file = st.file_uploader("Choose a PDF or Image file", type=["pdf", "jpg", "jpeg", "png"])
    if uploaded_file:
        file_name = uploaded_file.name
        if file_name.endswith('.pdf'):
            text = read_pdf(uploaded_file)
        elif file_name.endswith(('.jpg', '.jpeg', '.png')):
            text = ocr_image(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a PDF or an image.")
            text = None
    else:
        text = None
elif input_option == "Paste Text":
    text = st.text_area("Paste text here", height=200)
else:
    url = st.text_input("Enter Website URL (e.g., Wikipedia page)")
    if url:
        text = fetch_website_content(url)
    else:
        text = None

if st.button("Analyze & Summarize", key="analyze_button"):
    if not text:
        st.warning("Please provide some text to summarize.")
    else:
        try:
            detected_language = langdetect.detect(text)
            language_code = 'ar' if detected_language == 'ar' else 'en'
            
            with st.spinner('Processing...'):
                # Summarization
                summary = summarize_text(
                    text, 
                    num_sentences=summary_length,
                    language=language_code,
                    use_transformer=use_transformer
                )
                
                st.subheader("📝 Summary")
                st.markdown(f'<div class="arabic-text">{summary}</div>' if language_code == 'ar' else summary, 
                           unsafe_allow_html=True)
                
                # Translation
                if auto_translate:
                    translated_summary = translate_text(
                        summary, 
                        src=language_code, 
                        dest='en' if language_code == 'ar' else 'ar'
                    )
                    st.subheader("🌍 Translated Summary")
                    st.markdown(f'<div class="arabic-text">{translated_summary}</div>' 
                               if (language_code == 'en' and auto_translate) else translated_summary, 
                               unsafe_allow_html=True)
                
                # Sentiment Analysis
                polarity, subjectivity = analyze_sentiment(text)
                st.subheader("😊 Sentiment Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Polarity", f"{polarity:.2f}", help="-1 = Negative, 0 = Neutral, 1 = Positive")
                with col2:
                    st.metric("Subjectivity", f"{subjectivity:.2f}", help="0 = Objective, 1 = Subjective")
                
                # Keywords
                keywords = extract_keywords(text)
                st.subheader("🔑 Keywords")
                st.write(" ".join([f"`{kw}`" for kw in keywords]))
                
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
