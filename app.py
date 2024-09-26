import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import torch.nn.functional as F
import langdetect
from langdetect.lang_detect_exception import LangDetectException
import warnings
import time

app = FastAPI()

# Suppress warning
warnings.filterwarnings("ignore", message="The sentencepiece tokenizer that you are converting to a fast tokenizer uses the byte fallback option which is not implemented in the fast tokenizers.", category=UserWarning)

# Pydantic model for request body
class TextData(BaseModel):
    texts: list[str]

def clean_text(text):
    """
    Clean the input text by converting to lowercase, removing URLs, emails, and non-alphanumeric characters.
    """
    text = text.lower()
    return text

# Pipeline version for dynamically loading models
def load_model_english():
    """
    Dynamically load the English hate speech detection model.
    """
    return pipeline("text-classification", model="Hate-speech-CNERG/dehatebert-mono-english")

def load_model_tagalog():
    """
    Dynamically load the Tagalog hate speech detection model.
    """
    return pipeline("text-classification", model="ggpt1006/tl-hatespeech-detection")

def predict_filipino(text):
    """
    Predict if the given Tagalog text is hate speech using the dynamically loaded model.
    """
    model = load_model_tagalog()  # Load model dynamically
    start_time = time.time()  # Start timing
    
    result = model(text)
    prediction = result[0]['label']
    confidence = result[0]['score']
    
    end_time = time.time()  # End timing
    inference_time = end_time - start_time  # Calculate inference time

    return prediction, confidence, inference_time

def predict_english(text):
    """
    Predict if the given English text is hate speech using the dynamically loaded model.
    """
    model = load_model_english()  # Load model dynamically
    start_time = time.time()  # Start timing

    result = model(text)
    prediction = result[0]['label']
    confidence = result[0]['score']
    
    end_time = time.time()  # End timing
    inference_time = end_time - start_time  # Calculate inference time

    return prediction, confidence, inference_time

@app.get('/')
def index():
    """
    Root endpoint to check if the service is running.
    """
    return {'message': 'Running'}

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

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
