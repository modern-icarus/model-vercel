import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import langdetect
from langdetect.lang_detect_exception import LangDetectException
import warnings
import time

app = FastAPI()

# Suppress warning
warnings.filterwarnings("ignore", message="The sentencepiece tokenizer that you are converting to a fast tokenizer uses the byte fallback option which is not implemented in the fast tokenizers.", category=UserWarning)

# English tokenizer and model
english_tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/dehatebert-mono-english")
english_model = AutoModelForSequenceClassification.from_pretrained("Hate-speech-CNERG/dehatebert-mono-english")

# Tagalog tokenizer and model
tagalog_tokenizer = AutoTokenizer.from_pretrained("ggpt1006/tl-hatespeech-detection")
tagalog_model = AutoModelForSequenceClassification.from_pretrained("ggpt1006/tl-hatespeech-detection")

# Pydantic model for request body
class TextData(BaseModel):
    texts: list[str]

def clean_text(text):
    """
    Clean the input text by converting to lowercase characters.
    """
    text = text.lower()
    return text

def predict_filipino(text):
    """
    Predict if the given Tagalog text is hate speech using the RoBERTa model.
    """
    start_time = time.time()  # Start timing

    inputs = tagalog_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = tagalog_model(**inputs)
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=-1)
    confidence, prediction = torch.max(probabilities, dim=-1)

    end_time = time.time()  # End timing
    inference_time = end_time - start_time  # Calculate inference time

    return prediction.item(), confidence.item(), inference_time

def predict_english(text):
    """
    Predict if the given English text is hate speech using the transformer model.
    """
    start_time = time.time()  # Start timing

    inputs = english_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = english_model(**inputs)
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=-1)
    confidence, prediction = torch.max(probabilities, dim=-1)

    end_time = time.time()  # End timing
    inference_time = end_time - start_time  # Calculate inference time

    return prediction.item(), confidence.item(), inference_time

@app.get('/')
def index():
    """
    Root endpoint to check if the service is running.
    """
    return {'message': 'Welcome'}

@app.post('/predict')
async def predict_text(text_data: TextData):
    """
    Endpoint to predict hate speech in the given list of texts.
    """
    predictions = []
    for text in text_data.texts:
        cleaned_text = clean_text(text)
        try:
            language = langdetect.detect(cleaned_text)
            if language == 'en':
                prediction, confidence, inference_time = predict_english(cleaned_text)
            else:
                prediction, confidence, inference_time = predict_filipino(cleaned_text)
        except LangDetectException:
            # Default to Tagalog if language detection fails
            prediction, confidence, inference_time = predict_filipino(cleaned_text)
        
        predictions.append({
            'prediction': prediction,
            'confidence': confidence,
            'inference_time': inference_time
        })
    
    return {'predictions': predictions}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
