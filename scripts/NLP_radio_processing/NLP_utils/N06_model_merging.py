# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: nomarker
#       format_version: '1.0'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: f1_strat_manager
#     language: python
#     name: python3
# ---

# # Model Merging
#
# I am now at the end of the pipeline that I defined in `N04_radio_info.ipynb`:
#
# $Radio Message → [Sentiment Analysis] → [Intent Classification] → [Entity Extraction] → Structured Data$
#
# The Structured data is neccesary to output a **standarized JSON format** that would be used by the logical agent in week 5 planning, alongside other models.
#
# ## Objetives of this notebook
#
# The objectives and task to be done in this notebook are the following ones:
#
# 1. *Configure the mmodel paths*: first step, where I will configure the structure of the JSON.
#
# 2. *Load the pretrained models*: I will implement a function that will load the three elected models for making the radio analysis.
#
# 3. *Prediction functions*: for each model, it is necessary to implement a function where each models make their predictions.
#
# 4. *Audio Transcription*: first example usage, where I´ll use again Whisper for transcribing the radio message. Important for the urute if we want to directly pass the agent radio messages by radio.
#
# 5. *Integrating the pipeline into a single function*: this function will make the predictions of the three models.
#
# 6. *Analyzing text or audio examples*: two cells are going to be implemented for making the predictions in different types of radio formats.
#

# ---

# ## 1. Import Necessary Libraries

from datetime import datetime
import json
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
import spacy
import whisper
import torch
from transformers import BertForTokenClassification, BertConfig, BertTokenizer
import warnings

import logging
import transformers
# Ignorar FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
transformers.logging.set_verbosity_error()


# ## 2. Configuration
#
# In this cell, I will configure the format of the JSON. It is necessary to specify:
#
# - The model paths:
#     - ``../../outputs/week4/models/best_roberta_sentiment_model.pt``: best sentiment model developed.
#     - ``../../outputs/week4/models/best_roberta_large_intent_model.pt``: best intent model developed.
#     - ``../../outputs/week4/models/best_focused_bert_model.pt``: best NER model.
#     - `Whisper turbo` model.
#
# - Label mapppings of the models:
#     - **sentiment**: positive, negative, neutral.
#     - **intention**: INFORMATION, PROBLEM, ORDER, STRATEGY, WARNING, QUESTION.
#     - **NER**: ACTION, SITUATION, INCIDENT, STRATEGY_INSTRUCTION, POSITION_CHANGE, PIT_CALL, TRACK_CONDITION, TECHNICAL_ISSUE, WEATHER.
#
# - Demo input for making the correct structure.

# Configure model paths and settings - adjusted to match your specific models
CONFIG = {
    # Model paths
    "sentiment_model_path": "../../outputs/week4/models/best_roberta_sentiment_model.pt",
    "intent_model_path": "../../outputs/week4/models/best_roberta_large_intent_model.pt",
    "ner_model_path": "../../outputs/week4/models/best_focused_bert_model.pt",
    "whisper_model_size": "turbo",  # Using Whisper turbo

    # Label mappings
    "sentiment_labels": ["positive", "negative", "neutral"],
    "intent_labels": ["INFORMATION", "PROBLEM", "ORDER",  "WARNING", "QUESTION"],

    # Entity mapping (SpaCy entity labels → output format)
    "entity_mapping": {
        "B-ACTION": "action",
        "I-ACTION": "action",
        "B-INCIDENT": "incident",
        "I-INCIDENT": "incident",
        "B-PIT_CALL": "pit",
        "I-PIT_CALL": "pit",
        "B-POSITION_CHANGE": "position",
        "I-POSITION_CHANGE": "position",
        "B-SITUATION": "situation",
        "I-SITUATION": "situation",
        "B-STRATEGY_INSTRUCTION": "strategy",
        "I-STRATEGY_INSTRUCTION": "strategy",
        "B-TECHNICAL_ISSUE": "technical",
        "I-TECHNICAL_ISSUE": "technical",
        "B-TRACK_CONDITION": "track",
        "I-TRACK_CONDITION": "track",
        "B-WEATHER": "weather",
        "I-WEATHER": "weather",
        "O": "O"
    },

    # Demo inputs
    "example_text": "Box this lap for softs, Hamilton is catching up"
}


# ---

# ## 3. Loading Sentiment Model

def load_sentiment_model(device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Load the sentiment analysis model

    Args:
        device: Device to load the model on ('cuda' or 'cpu')

    Returns:
        Dictionary containing the loaded model and tokenizer
    """
    print(f"Loading sentiment model on {device}...")

    # Define base model for sentiment analysis
    base_model = "roberta-base"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(CONFIG["sentiment_labels"])
    )

    # Load fine-tuned weights
    model.load_state_dict(torch.load(
        CONFIG["sentiment_model_path"], map_location=device))

    # Move model to specified device and set to evaluation mode
    model.to(device)
    model.eval()

    print("Sentiment model loaded successfully!")

    return {
        "tokenizer": tokenizer,
        "model": model
    }


# ---

# ## 4. Loading Intent Model

def load_intent_model(device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Load the intent classification model

    Args:
        device: Device to load the model on ('cuda' or 'cpu')

    Returns:
        Dictionary containing the loaded model and tokenizer
    """
    print(f"Loading intent classification model on {device}...")

    # Define base model for intent classification
    base_model = "roberta-large"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(CONFIG["intent_labels"])
    )

    # Load fine-tuned weights
    model.load_state_dict(torch.load(
        CONFIG["intent_model_path"], map_location=device))

    # Move model to specified device and set to evaluation mode
    model.to(device)
    model.eval()

    print("Intent classification model loaded successfully!")

    return {
        "tokenizer": tokenizer,
        "model": model
    }


# ---

# ## 5. Loading NER model

def load_bert_ner_model(CONFIG, device="cuda" if torch.cuda.is_available() else "cpu"):

    # Create an initial congiguration based on the pretrained model, but with 19 labels
    config = BertConfig.from_pretrained(
        "dbmdz/bert-large-cased-finetuned-conll03-english",
        num_labels=len(CONFIG["entity_mapping"])
    )

    # Initilialize the model with this cofiguration
    model = BertForTokenClassification(config)

    # Load the weights of trained checkpoint with 19 labels
    model.load_state_dict(torch.load(
        CONFIG["ner_model_path"], map_location=device))
    model.to(device)
    model.eval()

    # Load the tokenizer of the pretrained model
    tokenizer = BertTokenizer.from_pretrained(
        "dbmdz/bert-large-cased-finetuned-conll03-english")

    print("NER classification model loaded successfully!")
    return {
        "tokenizer": tokenizer,
        "model": model
    }


# ---

# ## 5. Sentiment Prediction Function

def predict_sentiment(text, model, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Predicts the sentiment of an F1 radio message.

    Args:
        text (str): Text of the radio message.
        model: Sentiment model.
        tokenizer: Tokenizer for preprocessing the text.
        device: Device for inference.

    Returns:
        dict: Dictionary with text, sentiment, and confidence.
    """
    # Tokenize the text
    encoded_text = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )

    # Move inputs to the device
    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        pred_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][pred_class].item()

    # Map prediction to sentiment
    sentiments = ['positive', 'neutral', 'negative']

    # Return result as a dictionary
    result = {
        "text": text,
        "sentiment": sentiments[pred_class],
        "confidence": round(confidence * 100, 2)  # Convert to percentage
    }

    return result


# ---

# ## 6. Intent Prediction Function

def predict_intent(text, model, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Predicts the intent of an F1 radio message.

    Args:
        text (str): Text of the radio message.
        model: Intent classification model.
        tokenizer: Tokenizer for preprocessing the text.
        device: Device for inference.

    Returns:
        dict: Dictionary with text, predicted intent, and confidence.
    """
    # Tokenize the text
    encoded_text = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )

    # Move inputs to the device
    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        pred_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][pred_class].item()

    # Map prediction to intent (assuming you have defined this in your CONFIG)
    intents = CONFIG["intent_labels"]

    # Return result as a dictionary
    result = {
        "text": text,
        "intent": intents[pred_class],
        "confidence": round(confidence * 100, 2)  # Convert to percentage
    }

    return result


def analyze_radio_message(radio_message):
    """
    Analyzes a radio message using sentiment, intent, and NER models.

    Args:
        radio_message (str): The radio message to analyze.

    Returns:
        str: The path to the generated JSON file.
    """
    import sys
    from io import StringIO

    # Redirect stdout to suppress model loading messages
    original_stdout = sys.stdout
    sys.stdout = StringIO()

    # Load sentiment model
    sentiment_result = load_sentiment_model()
    sentiment_model = sentiment_result["model"]
    sentiment_tokenizer = sentiment_result["tokenizer"]

    # Load intent model
    intent_result = load_intent_model()
    intent_model = intent_result["model"]
    intent_tokenizer = intent_result["tokenizer"]

    # Load NER model
    ner_result = load_bert_ner_model(CONFIG)
    ner_model = ner_result["model"]
    ner_tokenizer = ner_result["tokenizer"]

    # Execute NER notebook (assuming this is needed)
    from .N05_ner_models import analyze_f1_radio

    sys.stdout = original_stdout  # Restore stdout

    # Get predictions
    sentiment_prediction = predict_sentiment(
        radio_message, sentiment_model, sentiment_tokenizer)
    intent_prediction = predict_intent(
        radio_message, intent_model, intent_tokenizer)
    ner_entities = analyze_f1_radio(radio_message)

    # Structure response as a dictionary
    response = {
        "message": radio_message,
        "analysis": {
            "sentiment": sentiment_prediction["sentiment"],
            "intent": intent_prediction["intent"],
            "entities": ner_entities
        }
    }

    # Create a unique filename based on current date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    json_filename = f"../../outputs/week4/json/radio_analysis_{timestamp}.json"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(json_filename), exist_ok=True)

    # Save response as a JSON file
    with open(json_filename, "w") as json_file:
        json.dump(response, json_file, indent=2)

    print(f"Analysis saved to {json_filename}")

    return json_filename  # Return the filename for reference


def transcribe_audio(audio_path):
    """
    Transcribe F1 team radio audio using the Whisper model.

    Args:
        audio_path (str): Path to the audio file

    Returns:
        str: Transcribed text
    """
    print("Loading Whisper model...")
    # You can use "tiny", "base", "small", "medium", or "large"
    model = whisper.load_model("turbo")

    print(f"Transcribing audio: {audio_path}")
    result = model.transcribe(audio_path)

    print("Transcription completed.")
    return result["text"]


# # Path to example audio file
# audio_path = "../../f1-strategy/data/audio/driver_(14,)/driver_(14,)_monaco_radio_168.mp3"


# # 🏁 NLP Pipeline Summary: Model Integration
#
# ## What We've Accomplished
#
# In this notebook, we successfully built a complete NLP pipeline for analyzing F1 team radio messages, completing the workflow:
#
# `Radio Message → [Sentiment Analysis] → [Intent Classification] → [Entity Extraction] → Structured Data`
#
# ## Key Components Implemented
#
# 1. **Model Configuration**:
#    - Defined paths for our pretrained models
#    - Configured label mappings for sentiment, intent, and entity classes
#    - Set up a standardized JSON output format
#
# 2. **Model Loading Functions**:
#    - Created functions to load all three specialized models:
#      - RoBERTa-base for sentiment analysis
#      - RoBERTa-large for intent classification
#      - BERT-large for named entity recognition (NER)
#
# 3. **Prediction Functions**:
#    - Implemented dedicated functions for each analysis type
#    - Created standardized output formats for each model
#
# 4. **Integration Pipeline**:
#    - Combined all models into a single `analyze_radio_message()` function
#    - Created a workflow that processes raw text through all models
#    - Included Whisper transcription support for audio inputs
#
# 5. **JSON Output**:
#    - Developed a structured output format containing:
#      - Original message
#      - Sentiment (positive, negative, neutral)
#      - Intent (INFORMATION, PROBLEM, ORDER, WARNING, QUESTION)
#      - Named entities with their types and positions
#
# ## Next Steps
#
# This pipeline is now ready to be integrated with the logical agent system from week 5, providing rich structured data from team radio communications that can be used for strategic decision-making in our F1 strategy system.
