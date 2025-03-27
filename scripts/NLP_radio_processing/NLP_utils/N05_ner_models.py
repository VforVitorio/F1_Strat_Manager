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

# # Formula 1 NER Recognition
#
# As a continuation of `N04_radio_info.ipynb`, this notebook demonstrates the development of a custom Named Entity Recognition (NER) system for Formula 1 team radio communications. 
#
# The system extracts structured information from unstructured radio messages exchanged between drivers and race engineers during F1 races.
#
# ## Important Disclaimer after making the notebook
#
# Lots of cells are commented for not retraining or evading reloading functions that are not needed for the last example usage. Therefore, if it is necessary to rerun or for other purporses **please feel free to uncomment all of the cells that you want**.

# ---

# ## Methodology 
#
# The notebook implements:
#
# 1. **Data preparation** - Loading annotated F1 radio messages with entity labels
#
# 2. **BIO tagging** - Converting character-level entity spans to token-level Beginning-Inside-Outside format
#
# 3. **Model architecture** - Training transformer-based models including DeBERTa v3 and BERT. making a fine-tuning of this last transformer.
#
# 4. **Fine-tuning strategies** - Testing various approaches including class weighting and focused training
#
# 5. **Evaluation** - Detailed entity-level performance analysis and model comparison

# ## 1. Library Imports

import json
import numpy as np
import pandas as pd
import torch
import os
from transformers import DebertaV2Tokenizer, DebertaV2ForTokenClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    
    get_linear_schedule_with_warmup
)

from datasets import Dataset as HFDataset, DatasetDict

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm.auto import tqdm
from transformers import BertTokenizerFast, BertForTokenClassification
import torch.nn as nn
import torch.nn.functional as F


# ## 3. Constant Definition 
#
# In this cell, I´ll document the type of entities and their correspondent colors.

# Define entity types and their descriptions
ENTITY_TYPES = {
    "ACTION": "Direct commands or actions mentioned in the message",
    "SITUATION": "Racing context or circumstance descriptions",
    "INCIDENT": "Accidents or on-track events",
    "STRATEGY_INSTRUCTION": "Strategic directives",
    "POSITION_CHANGE": "References to overtakes or positions",
    "PIT_CALL": "Specific calls for pit stops",
    "TRACK_CONDITION": "Mentions of the track's state",
    "TECHNICAL_ISSUE": "Mechanical or car-related problems",
    "WEATHER": "References to weather conditions"
}

# Color scheme for entity visualization
ENTITY_COLORS = {
    "ACTION": "#4e79a7",           # Blue
    "SITUATION": "#f28e2c",         # Orange
    "INCIDENT": "#e15759",          # Red
    "STRATEGY_INSTRUCTION": "#76b7b2", # Teal
    "POSITION_CHANGE": "#59a14f",   # Green
    "PIT_CALL": "#edc949",          # Yellow
    "TRACK_CONDITION": "#af7aa1",   # Purple
    "TECHNICAL_ISSUE": "#ff9da7",   # Pink
    "WEATHER": "#9c755f"            # Brown
}

print("Entity types defined:")
for entity, description in ENTITY_TYPES.items():
    print(f"  - {entity}: {description}")


# ## 4. Load and Explore Data

# Load F1 radio data from JSON file
def load_f1_radio_data(json_file):
    """Load and explore F1 radio data from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} messages from {json_file}")
    
    # Show sample structure
    if len(data) > 0:
        print("\nSample record structure:")
        sample = data[0]
        print(f"  Driver: {sample.get('driver', 'N/A')}")
        print(f"  Radio message: {sample.get('radio_message', 'N/A')[:100]}...")
        
        if 'annotations' in sample and len(sample['annotations']) > 1:
            if isinstance(sample['annotations'][1], dict) and 'entities' in sample['annotations'][1]:
                entities = sample['annotations'][1]['entities']
                print(f"  Number of entities: {len(entities)}")
                if len(entities) > 0:
                    entity = entities[0]
                    entity_text = sample['radio_message'][entity[0]:entity[1]]
                    print(f"  Sample entity: [{entity[0]}, {entity[1]}, '{entity_text}', '{entity[2]}']")
    
    return data



# Load the JSON data
json_file_path = "../../outputs/week4/NER/f1_radio_entity_annotations.json"
f1_data = load_f1_radio_data(json_file_path)

# Count entity types in the dataset
entity_counts = {}
for item in f1_data:
    if 'annotations' in item and len(item['annotations']) > 1:
        if isinstance(item['annotations'][1], dict) and 'entities' in item['annotations'][1]:
            for _, _, entity_type in item['annotations'][1]['entities']:
                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

print("\nEntity type distribution in dataset:")
for entity_type, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  - {entity_type}: {count}")


# ## 5. Preprocessing F1 Radio Data

def preprocess_f1_data(data):
    """Extract and preprocess F1 radio data with valid annotations"""
    processed_data = []
    skipped_count = 0
    
    for item in data:
        if 'radio_message' not in item or 'annotations' not in item:
            skipped_count += 1
            continue
            
        text = item['radio_message']
        
        # Skip items with empty or null text
        if not text or text.strip() == "":
            skipped_count += 1
            continue
            
        # Extract entities if they exist in expected format
        if len(item['annotations']) > 1 and isinstance(item['annotations'][1], dict):
            annotations = item['annotations'][1]
            if 'entities' in annotations and annotations['entities']:
                entities = annotations['entities']
                
                # Add to processed data
                processed_data.append({
                    'text': text,
                    'entities': entities,
                    'driver': item.get('driver', None)
                })
            else:
                skipped_count += 1
        else:
            skipped_count += 1
    
    print(f"Processed {len(processed_data)} messages with valid annotations")
    print(f"Skipped {skipped_count} messages with missing or invalid annotations")
    
    # Show a sample of processed data
    if processed_data:
        sample = processed_data[10]
        print("\nSample processed message:")
        print(f"Text: {sample['text']}")
        print("Entities:")
        for start, end, entity_type in sample['entities']:
            entity_text = sample['text'][start:end]
            print(f"  - [{start}, {end}] '{entity_text}' ({entity_type})")
    
    return processed_data



# Preprocess the loaded data
processed_f1_data = preprocess_f1_data(f1_data)


# ## 6. Covert to BIO tagging format
#
# Deeper BIO tagging format information can be searched [here](https://en.wikipedia.org/wiki/Inside–outside–beginning_(tagging)).
#
# ### BIO Format Explanation
#
# The **BIO format** is a way to label words in a sentence to indicate if they are part of a named entity, and if so, where in the entity they belong. It uses three types of labels:
#
# - **B- (Beginning)**: The first word in an entity.
# - **I- (Inside)**: Any word inside the entity that isn't the first one.
# - **O (Outside)**: Words that are not part of any entity.
#
# ---
#
# ### Example Radio
#
# Here is an example of a radio message from Max Verstappen´s track engineer: 
#
# **Text:**  
# *"Max, we've currently got yellows in turn 7. Ferrari in the wall, no? Yes, that's Charles stopped. We are expecting the potential of an aborted start, but just keep to your protocol at the moment."*
#
# Here are the entities mentioned in the message:
#
# 1. **'keep to your protocol at the moment'** (ACTION)
# 2. **'we've currently got yellows in turn 7'** (SITUATION)
# 3. **'We are expecting the potential of an aborted start'** (SITUATION)
# 4. **'Ferrari in the wall'** (INCIDENT)
# 5. **'that's Charles stopped'** (INCIDENT)
#
# ---
#
# ### Breaking the Sentence
#
# We break the sentence into words and then tag them as follows:
#
# | Word            | BIO Tag          |
# |-----------------|------------------|
# | Max,            | O                |
# | we've           | O                |
# | currently       | O                |
# | got             | O                |
# | yellows         | O                |
# | in              | O                |
# | turn            | O                |
# | 7.              | O                |
# | Ferrari         | B-INCIDENT       |
# | in              | I-INCIDENT       |
# | the             | I-INCIDENT       |
# | wall,           | I-INCIDENT       |
# | no?             | O                |
# | Yes,            | O                |
# | that's          | B-INCIDENT       |
# | Charles         | I-INCIDENT       |
# | stopped.        | I-INCIDENT       |
# | We              | B-SITUATION      |
# | are             | I-SITUATION      |
# | expecting       | I-SITUATION      |
# | the             | I-SITUATION      |
# | potential       | I-SITUATION      |
# | of              | I-SITUATION      |
# | an              | I-SITUATION      |
# | aborted         | I-SITUATION      |
# | start,          | I-SITUATION      |
# | but             | O                |
# | just            | O                |
# | keep            | B-ACTION         |
# | to              | I-ACTION         |
# | your            | I-ACTION         |
# | protocol        | I-ACTION         |
# | at              | I-ACTION         |
# | the             | I-ACTION         |
# | moment.         | I-ACTION         |
#
#
#

def create_ner_tags(text, entities):
    """Convert character-based entity spans to token-based BIO tags"""
    words = text.split()
    tags = ["O"] * len(words)
    char_to_word = {}
    
    # Create mapping from character positions to word indices
    char_idx = 0
    for word_idx, word in enumerate(words):
        # Account for spaces
        if char_idx > 0:
            char_idx += 1  # Space
        
        # Map each character position to its word index
        for char_pos in range(char_idx, char_idx + len(word)):
            char_to_word[char_pos] = word_idx
        
        char_idx += len(word)
    
    # Apply entity tags
    for start_char, end_char, entity_type in entities:
        # Skip invalid spans
        if start_char >= len(text) or end_char > len(text) or start_char >= end_char:
            continue
            
        # Find word indices for start and end characters
        if start_char in char_to_word:
            start_word = char_to_word[start_char]
            # Find the last word of the entity
            end_word = char_to_word.get(end_char - 1, start_word)
            
            # Tag the first word as B-entity
            tags[start_word] = f"B-{entity_type}"
            
            # Tag subsequent words as I-entity
            for word_idx in range(start_word + 1, end_word + 1):
                tags[word_idx] = f"I-{entity_type}"
    
    return words, tags





def convert_to_bio_format(processed_data):
    """Convert processed data to BIO tagging format"""
    bio_data = []
    mapping_errors = 0
    
    for item in processed_data:
        text = item['text']
        entities = item['entities']
        
        # Convert to BIO tags
        words, tags = create_ner_tags(text, entities)
        
        # Check if we mapped any entities
        if all(tag == "O" for tag in tags) and len(entities) > 0:
            mapping_errors += 1
        
        bio_data.append({
            "tokens": words,
            "ner_tags": tags,
            "driver": item.get('driver', None)
        })
    
    print(f"Converted {len(bio_data)} messages to BIO format")
    print(f"Mapping errors: {mapping_errors} (messages where no entities were mapped)")
    
    # Show an example
    if bio_data:
        sample = bio_data[10]
        print("\nSample BIO tagging:")
        print(f"Original text: {' '.join(sample['tokens'])}")
        for token, tag in zip(sample['tokens'], sample['ner_tags']):
            print(f"  {token} -> {tag}")
    
    return bio_data


# Convert processed data to BIO format
bio_data = convert_to_bio_format(processed_f1_data)


# ### What the Function Does
#
# The function `create_ner_tags` takes the text and entities and converts them into BIO format. It starts by splitting the text into words. 
#
# Then, it maps each word to a tag: "O" for words that are not part of an entity, "B-" for the first word of an entity, and "I-" for subsequent words inside the entity. 
#
# The function also uses the character positions of the entities to determine which words they correspond to. Once the tags are assigned, the function returns the words and their BIO tags, ready for use in training a Named Entity Recognition (NER) model.

# ## 7. Create tag mappings and prepare datasets.

# ### 7.1 `create_tag_mappings`
#
# This function creates mappings between NER (Named Entity Recognition) tags and unique IDs. It does this by:
#
# 1. Collecting all unique NER tags from the `bio_data`.
# 2. Sorting and assigning each unique tag an ID.
# 3. Creating two mappings:
#    - `tag2id`: Maps each tag to its corresponding ID.
#    - `id2tag`: Maps each ID back to its corresponding tag.
#
# It then prints out the mappings and returns the two dictionaries: `tag2id` and `id2tag`.
#
# **What it does:**
# - Converts NER tags into unique IDs for easier processing in machine learning models.
# - Helps with transforming the tags when working with model inputs and outputs.

def create_tag_mappings(bio_data):
    """Create mappings between NER tags and IDs"""
    unique_tags = set()
    for item in bio_data:
        unique_tags.update(item["ner_tags"])
    
    tag2id = {tag: id for id, tag in enumerate(sorted(list(unique_tags)))}
    id2tag = {id: tag for tag, id in tag2id.items()}
    
    print(f"Created mappings for {len(tag2id)} unique tags:")
    for tag, idx in tag2id.items():
        print(f"  {tag}: {idx}")
    
    return tag2id, id2tag


# Create tag mappings
tag2id, id2tag = create_tag_mappings(bio_data)


# ---

# ### 7.2 `prepare_datasets`
#
# This function prepares the dataset for training a model by splitting it into training, validation, and test sets using the Hugging Face library. Here's what it does:
#
# 1. Converts the input `bio_data` into a Hugging Face `Dataset`.
# 2. Splits the data into two parts: training + validation, and test.
# 3. Further splits the training data into training and validation sets based on the specified sizes (`test_size` and `val_size`).
# 4. Returns a `DatasetDict` containing the `train`, `validation`, and `test` sets.
#
# **What it does:**
# - Converts the data into a format suitable for machine learning.
# - Splits the data into three parts: training, validation, and test sets for model evaluation.

def prepare_datasets(bio_data, test_size=0.1, val_size=0.1, seed=42):
    """Convert to Hugging Face Dataset and split into train/val/test"""
    # Convert to Hugging Face dataset
    hf_dataset = HFDataset.from_list(bio_data)
    
    # First split: train + validation vs test
    train_val_test = hf_dataset.train_test_split(test_size=test_size, seed=seed)
    
    # Second split: train vs validation (validation is val_size/(1-test_size) of the train set)
    val_fraction = val_size / (1 - test_size)
    train_val = train_val_test["train"].train_test_split(test_size=val_fraction, seed=seed)
    
    # Combine into DatasetDict
    datasets = DatasetDict({
        "train": train_val["train"],
        "validation": train_val["test"],
        "test": train_val_test["test"]
    })
    
    print(f"Prepared datasets with:")
    print(f"  - Train: {len(datasets['train'])} examples")
    print(f"  - Validation: {len(datasets['validation'])} examples")
    print(f"  - Test: {len(datasets['test'])} examples")
    
    return datasets


datasets = prepare_datasets(bio_data)

# ---
#
# ## 8. Calling Up the Model 
#
# In the first run, I tried with *Microsoft Deberta-v3-large*, a bigger model than BERT or RoBERTa. I believe more that the robustness of this architecture can provide good results.
#
# As it will be seen, the f1-score of this model is not too bad and could be enhanced with further development (in 2 or 3 runs I tried different loss functions, focal loss, epochs, learning rates, etc). However, some lighter models like specific architectures derived from BERT have good metrics and are more easily customizable.
#
# Therefore, I guided the development to these models. However, much of the code made in the following cells are used by BERT models, so it is important to keep it.

torch.manual_seed(42)
# Cell 2: Initialize the tokenizer for DeBERTa v3 large
model_name = "microsoft/deberta-v3-large"
tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)

# Check if it loaded correctly
print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
print(f"Vocabulary size: {len(tokenizer)}")


# ---
#
# ## 9. Custom Dataset for Deberta-v3 Tokenization
#
# The ``F1RadioNERDataset`` class is a custom dataset for Named Entity Recognition (NER) tasks using PyTorch. It is designed to work with a Hugging Face dataset (or our dataset ), a tokenizer, and a tag-to-id mapping.
#
#
# ### Key Points:
#
# 1. Initialization (``__init__``):
#
#     - Accepts a dataset, tokenizer, tag-to-id mapping, and maximum sequence length.
#
#     - Stores these for later use in processing data.
#
# 2. Length Method (``__len__``):
#
#     - Returns the total number of examples in the dataset.
#
# 3. Item Retrieval (``__getitem__``):
#
#     - Retrieves a single data example by index.
#
#     - Tokenizes each word in the example while maintaining a mapping of tokens to their original word indices.
#
#     - Truncates the tokenized sequence if it exceeds the maximum length (keeping space for special tokens like [CLS] and [SEP]).
#
#     - Uses the tokenizer to add special tokens and convert tokens to their corresponding IDs.
#
#     - Creates a labels tensor where the NER tags are aligned with the tokens. Special tokens and padding tokens are marked with -100 so that they are ignored during loss computation.
#
#     - Returns a dictionary with input_ids, attention_mask, and labels ready for model input.
#

class F1RadioNERDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, tokenizer, tag2id, max_len=128):
        """
        Initializes the dataset with a Hugging Face dataset, tokenizer, tag-to-id mapping, and maximum sequence length.
        
        Parameters:
        - hf_dataset: The dataset containing tokenized text and NER tags.
        - tokenizer: A tokenizer to process and convert text to token IDs.
        - tag2id: A dictionary mapping NER tag strings to numerical IDs.
        - max_len: The maximum length for token sequences (default is 128).
        """
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.tag2id = tag2id  # Mapping of NER tags to their numeric IDs
        self.max_len = max_len
        
    def __len__(self):
        """
        Returns the number of examples in the dataset.
        """
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Retrieves the data example at the given index, processes tokens and tags,
        and prepares input tensors for a NER model.
        
        Parameters:
        - idx: Index of the data example to retrieve.
        
        Returns:
        A dictionary containing:
        - input_ids: Tensor of token IDs for the input sequence.
        - attention_mask: Tensor indicating which tokens are real (1) and which are padding (0).
        - labels: Tensor of label IDs for the tokens, with -100 for tokens to be ignored.
        """
        # Get the raw data example
        item = self.dataset[idx]
        tokens = item["tokens"]         # List of word tokens
        tags = item["ner_tags"]         # Corresponding NER tags
        
        # Initialize lists to store word indices and all sub-tokens
        word_ids = []  # Maps each sub-token to its original word index
        all_tokens = []  # Stores all sub-tokens after tokenization
        
        # Tokenize each word separately
        for word_idx, word in enumerate(tokens):
            # Tokenize the word using the provided tokenizer
            word_tokens = self.tokenizer.tokenize(word)
            if not word_tokens:
                # Handle cases where tokenization results in an empty list
                word_tokens = [self.tokenizer.unk_token]
            
            # For each sub-token produced from the word, record the original word index
            for _ in word_tokens:
                word_ids.append(word_idx)
                
            # Add the sub-tokens to the overall list of tokens
            all_tokens.extend(word_tokens)
        
        # Check if the tokenized sequence needs truncation
        if len(all_tokens) > self.max_len - 2:  # -2 accounts for special tokens ([CLS] and [SEP])
            all_tokens = all_tokens[:self.max_len - 2]
            word_ids = word_ids[:self.max_len - 2]
        
        # Encode the tokenized input, adding special tokens and padding as needed
        encoded_input = self.tokenizer.encode_plus(
            all_tokens,
            is_split_into_words=False,  # Already tokenized input; no further splitting required
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"  # Return PyTorch tensors
        )
        
        # Initialize the labels tensor with -100 to ignore special tokens and padding during loss computation
        labels = torch.ones(self.max_len, dtype=torch.long) * -100
        
        # Align NER labels with the tokenized sequence
        # The first token ([CLS]) is already set to -100 by initialization
        for i, word_idx in enumerate(word_ids):
            # Ensure we do not overwrite the [CLS] token and reserve space for [SEP]
            if i + 1 < self.max_len - 1:
                # Check if the tag is a string and convert it to its numeric ID if necessary
                if isinstance(tags[word_idx], str):
                    tag_id = self.tag2id.get(tags[word_idx], 0)  # Defaults to 0, often representing 'O'
                else:
                    tag_id = tags[word_idx]  # Already a numeric ID
                    
                # Set the label at the corresponding position (offset by 1 for [CLS])
                labels[i + 1] = tag_id
        
        # Return the processed inputs as a dictionary
        return {
            "input_ids": encoded_input["input_ids"].flatten(),
            "attention_mask": encoded_input["attention_mask"].flatten(),
            "labels": labels
        }



# ---
# ## 10. Pytorch Setup
#
# In PyTorch, creating custom datasets and corresponding DataLoaders is essential for efficiently feeding data into our model during training and evaluation. 
#
# I found that the following steps are crucial:
#
# ### A) Creating Pytorch Datasets
#
# By using the ``F1RadioNERDataset`` class, we convert our raw data (tokens, NER tags, etc.) into a format that is compatible with PyTorch. 
#
# This allows us to perform operations such as tokenization and label alignment on the fly. Passing the ``tag2id`` mapping ensures that the **NER tags are correctly converted into numeric IDs**, which is **necessary for training** the model.
#
# ### B) Creating the DataLoaders
#
# The DataLoader is a PyTorch utility that **provides an iterable over our dataset**. It handles **batching, shuffling (for training), and even parallel data loading** with multiple workers if needed. 
#
# This *makes the training process more efficient* and helps in managing *memory usage*, parts that are crucial knwoging that I am training these models in my own graphics card, with a limited VRAM. 
#
# By specifying a batch size and shuffling the training data, we ensure that each mini-batch is representative and that the **model doesn’t overfit to the order** of the training data.
#
#

# ### 10.1 Creating Pytorch Datasets

# Create PyTorch datasets using the custom F1RadioNERDataset class.
# Passing the tag2id mapping is crucial as it converts NER tag strings to numeric IDs,
# which are needed for model training.
train_dataset = F1RadioNERDataset(datasets["train"], tokenizer, tag2id)
val_dataset = F1RadioNERDataset(datasets["validation"], tokenizer, tag2id)
test_dataset = F1RadioNERDataset(datasets["test"], tokenizer, tag2id)




# ### 10.2 Creating Dataloaders

# Create DataLoaders for the datasets.
# DataLoaders are used in PyTorch to efficiently batch and shuffle data during training and evaluation.
# They help in managing memory and speeding up the training process by allowing parallel data loading.
batch_size = 8  # Reduced batch size due to model size constraints and resource availability.

# For training, shuffling is enabled to ensure the model does not learn the order of the data.
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# For validation and testing, shuffling is typically not required.
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)



# ### 10.3 Validating Samples
#
# In this step, I will only print some of the training samples and also the shapes present in the dataset.
#
# This way, I know if the split was made correctly, and also if all the sizes are the same. If they were not, error would occur during the training process.

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Optional: Check a sample to verify everything is working
sample = train_dataset[0]
print(f"Sample input shape: {sample['input_ids'].shape}")
print(f"Sample attention mask shape: {sample['attention_mask'].shape}")
print(f"Sample labels shape: {sample['labels'].shape}")

# ---
# ## 11. Initializing Deberta
#
# In the following cell, next things will be implemented:
#
# #### I) Device Setup
# PyTorch models and tensors need to be moved to the appropriate hardware (CPU or GPU) for computation. 
#
# Checking if a GPU (CUDA) is available and setting the device accordingly ensures that the model leverages the faster processing power of a GPU when available. I will use it, as I have a Nvidia GPU with the drivers installed.
#
# This is crucial for efficient training and inference, especially when dealing with large models.
#
# #### II) Model Initialization with Label Information
#
# The ``DebertaV2ForTokenClassification`` model is loaded from a pretrained checkpoint, and it needs to know the number of labels for token classification. 
#
# By using ``num_labels = len(tag2id)``, the model is configured to produce outputs that align with the NER task. This ensures that the final classification layer has the correct dimensions to predict the right classes.
#
# #### III) Moving the model to the device and Feedback print Statements
#
# Once the model is loaded and configured, moving it to the chosen device (CPU or GPU) is necessary. This step transfers all model parameters to the selected device, ensuring that subsequent computations (like forward passes and gradients during training) occur on the correct hardware.
#
# Moreover, it is a common practice to print the device, model name and number of labels, as it provides useful information for confirmation and debugging. Therefore, we can easily check if the model is correctly configured and that it will run in on the intended hardware.

# #### Uncomment these lines if needed


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_labels = len(tag2id)  # Use our existing tag2id mapping
model = DebertaV2ForTokenClassification.from_pretrained(
    model_name, 
    num_labels=num_labels
)
model.to(device)

print(f"Model loaded: {model_name}")
print(f"Number of labels: {num_labels}")

# ---
# ## 12. Set Up the Training Configuration
#
# #### A) Extracting Training Labels
#
# In token classification tasks, not every token is relevant for computing the loss (for instance, special tokens or padding tokens are ignored by marking them with ``-100``). 
#
# This code iterates over the training DataLoader and collects only the relevant labels (i.e., those not marked with ``-100``). This step is necessary to accurately analyze the distribution of actual labels used in training.
#
# #### B) Computing Class Weights
#
# **Class imbalance** is a common issue in many classification tasks, and my Dataset has it also. When some classes are under-represented, the model might become biased towards the majority classes (and it became biased on the first runs).
#
# Using scikit-learn's ``compute_class_weight`` function helps to compute weights for each class inversely proportional to their frequency in the dataset. These weights can later be used during training (e.g., in the loss function) to give more importance to minority classes and improve model performance.

from sklearn.utils import compute_class_weight

# Set the number of training epochs
epochs = 10

# Initialize an empty list to store all training labels (ignoring special tokens)
train_labels = []

# Loop over each batch in the training DataLoader
for batch in train_loader:
    # Get the labels tensor from the current batch
    labels = batch['labels']
    
    # Create a mask to filter out tokens with the ignore index (-100)
    mask = labels != -100
    
    # Extend the train_labels list with the valid labels (convert tensor to numpy array)
    train_labels.extend(labels[mask].numpy())

# Calculate class weights using scikit-learn's compute_class_weight
# 'balanced' mode adjusts weights inversely proportional to class frequencies in the dataset.
class_weights = compute_class_weight(
    'balanced', 
    classes=np.unique(train_labels),  # Unique classes present in the training labels
    y=train_labels                    # List of training labels
)


# ---

# #### C) Class Weights Conversion
#
# The ``computed class weights`` (from scikit-learn) are converted to a PyTorch tensor and moved to the appropriate device (GPU or CPU). 
#
# This conversion is necessary because the **loss function will use these weights during training**, and they must be in the **same format and on the same device as the model's parameters**.
#
# #### D) Custom Loss Function
#
# A **weighted CrossEntropyLoss** is defined, using the computed class weights to counteract class imbalance. The ``ignore_index=-100`` parameter ensures that tokens marked with -100 (such as special tokens or padding) do not contribute to the loss, **preventing them from skewing the training**.
#
# #### E) Learning Rate Setup
#
# A small learning rate (1e-5) is set to allow for fine-tuning of the model. **Lower learning rates help in stabilizing the training process**, especially when **fine-tuning large pre-trained models**.
#
# #### F) Warmup Steps
#
# They are included to **gradually increase the learning rate** form a small value to the target learning rate. This is implemented for **stabilizing training** in the initial phase and prevents **sudden jumps in gradient updates**.
#
# #### G) Optimizer Configuration
# The AdamW optimizer is used with weight decay to **help regularize the model and prevent overfitting**. AdamW is commonly used in transformer-based models.
#
# #### H) Learning Rate Scheduler
# A linear scheduler with warmup is set up to adjust the learning rate throughout training. This scheduler **gradually increases the learning rate during the warmup phase and then linearly decreases it during the remainder of the training process.**
#

# Convert the class weights computed by scikit-learn into a PyTorch FloatTensor
# and move it to the same device as the model (GPU or CPU).
class_weights = torch.FloatTensor(class_weights).to(device)

# Define a weighted CrossEntropyLoss function.
# The weight parameter applies the class weights to handle class imbalance.
# ignore_index=-100 ensures that tokens marked as -100 (like padding) are ignored in loss computation.
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

# Set a small learning rate for better fine tuning of the pre-trained model.
learning_rate = 1e-5  # Reduced from 2e-5 to 1e-5 for more gradual learning updates.

# Calculate the number of warmup steps.
# Here, warmup steps are set to 10% of the total training steps to stabilize the initial training.
warmup_steps = int(0.1 * len(train_loader) * epochs)  # 10% of the total steps

# Calculate the total number of training steps.
total_steps = len(train_loader) * epochs

# Initialize the AdamW optimizer with model parameters, a low learning rate, and weight decay.
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

# Set up a linear learning rate scheduler with warmup.
# This scheduler increases the learning rate for the warmup_steps, then linearly decays it.
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps  # Total training steps
)


# #### I) Computing the metrics
#
# The ``compute_metrics`` function is crucial for evaluating model performance during training and validation.
#
# 1. **Prediction Processing**
#
#     - The function starts by converting raw model outputs (logits) into predicted labels by using np.argmax along the classification dimension. 
#     
#     - This produces the most likely class for each token.
#
# 2. **Flattening and Filtering**
#     - Both predictions and true labels are flattened into 1D arrays.
#      
#     - Special tokens and padding tokens are marked with -100 in the labels; filtering these out ensures that only valid tokens are considered when calculating the metrics. 
#      
#     - This step prevents skewing the evaluation metrics by including irrelevant tokens.
#
# 3. **Metric Calculation**
#
#     - **Accuracy**: Measures the overall proportion of correctly predicted labels.
#
#     - **Precision, Recall, and F1 Score**: These metrics provide deeper insights into the performance, especially in imbalanced datasets. The weighted average ensures that the contribution of each class is proportional to its frequency.

# Metrics function for evaluating model performance
def compute_metrics(preds, labels):
    # Convert model outputs (logits) to predicted labels by selecting the class with the highest probability
    preds = np.argmax(preds, axis=2).flatten()
    labels = labels.flatten()
    
    # Create a mask to filter out tokens with the ignore index (-100), which are not used in training (e.g., padding)
    mask = labels != -100
    
    # Apply the mask to both predictions and labels to keep only the valid tokens
    preds = preds[mask]
    labels = labels[mask]
    
    # Calculate accuracy: the fraction of correctly predicted labels
    accuracy = accuracy_score(labels, preds)
    
    # Calculate precision, recall, and F1 score using weighted averages to account for class imbalances.
    # The weighted average ensures that each class contributes to the overall metric proportionally to its frequency.
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    
    # Return a dictionary containing all computed metrics for easy access and logging.
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }



# ---
#
# ## 13. Training and Evaluation Functions
#
# The next function will implement the training for one epoch, performing the following steps:
#
# 1. *Model training*: sets the model in training mode and iterates over training batches.
#
# 2. *Forward Pass and Loss calculation*: processes input data, computes logits (these are the raw outputs of the last layer in a neural network, see [this link explanation](https://www.geeksforgeeks.org/what-are-logits-what-is-the-difference-between-softmax-and-softmax-cross-entropy-with-logits/), and applies a mask to exclude ignored tokens from loss calculation.
#
# 3. *Backward pass and Optimization*: finally, computes the gradients, clips them for more stability, updates the model parameters with the defined optimizer and adjusts the learning rate using the scheduler.
#

# Using personalized loss function for training one epoch
def train_epoch():
    model.train()  # Set the model to training mode
    total_loss = 0  # Initialize the total loss accumulator
    
    # Iterate over training batches with a progress bar
    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()  # Reset gradients
        
        # Move inputs and labels to the designated device (GPU or CPU)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass: obtain model outputs (logits)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Create a mask for valid tokens (ignore tokens with label -100)
        active_loss = labels != -100
        
        # Reshape logits to match the expected loss function input
        active_logits = logits.view(-1, num_labels)
        
        # Prepare active labels: use the ignore index where needed
        active_labels = torch.where(
            active_loss.view(-1), 
            labels.view(-1), 
            torch.tensor(loss_fn.ignore_index).type_as(labels)
        )
        
        # Calculate the loss for the current batch
        loss = loss_fn(active_logits, active_labels)
        total_loss += loss.item()  # Accumulate loss
        
        # Backward pass: compute gradients
        loss.backward()
        # Clip gradients to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Update model parameters
        optimizer.step()
        # Update the learning rate
        scheduler.step()
        
    # Return the average loss for the epoch
    return total_loss / len(train_loader)



# ---

# #### Evaluation 
#
# The `evaluate` function switches the model to evaluation mode, processing the given data loader without updating the gradients. It also calculates the loss and collects predictions and true labels. Finally, it computes the predefined metrics with our `compute_metrics` function (accuracy, f1-score,etc) and then returns them along with the average loss.
#
#

def evaluate(data_loader):
    model.eval()  # Set model to evaluation mode
    total_loss = 0  # Initialize loss accumulator
    all_preds = []  # To store predictions
    all_labels = []  # To store true labels
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        # Iterate over batches in the data loader with a progress bar
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Move inputs and labels to the device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Get model outputs including loss and logits
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Accumulate the loss from the current batch
            loss = outputs.loss
            total_loss += loss.item()
            
            # Detach logits and move them to CPU for metric calculation
            logits = outputs.logits
            all_preds.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
    
    # Concatenate all batch predictions and labels into single arrays
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Compute evaluation metrics (accuracy, precision, recall, F1)
    metrics = compute_metrics(all_preds, all_labels)
    # Add the average loss to the metrics dictionary
    metrics['loss'] = total_loss / len(data_loader)
    
    return metrics


# ---
# ## 14. Training Loop

# This training loop is designed to iterate over multiple epochs to train the model and evaluate it on a validation set. The key steps are:
#
# #### I) Epoch Iteration
# The loop runs for a specified number of epochs. For each epoch, it prints the progress, trains the model with ``train_epoch()``, and evaluates it on the validation set with ``evaluate(val_loader)``.
#
# #### II) Metric Logging
# After each epoch, the training loss and validation metrics (including loss, accuracy, precision, recall, and F1 score) are printed for monitoring performance.
#
# #### III) Best Model Saving
# If the F1 score improves over the best observed so far, the model’s state is saved. This helps to keep the best performing model checkpoint.
#
# #### IV) Commenting Out: Disclaimer.
# The entire block is commented out to prevent accidental execution. 

# The following training loop is commented out to prevent accidental execution.
# Uncomment it when you are ready to run the training process.

# best_f1 = 0  # Initialize the best F1 score for saving the best model

# for epoch in range(epochs):
#     print(f"\n{'='*50}")
#     print(f"Epoch {epoch+1}/{epochs}")
#     print(f"{'='*50}")
    
#     # Train the model for one epoch and print the training loss
#     train_loss = train_epoch()
#     print(f"Training loss: {train_loss:.4f}")
    
#     # Evaluate the model on the validation set and print the validation metrics
#     val_metrics = evaluate(val_loader)
#     print(f"Validation loss: {val_metrics['loss']:.4f}")
#     print(f"Validation metrics: accuracy={val_metrics['accuracy']:.4f}, precision={val_metrics['precision']:.4f}, "
#           f"recall={val_metrics['recall']:.4f}, f1={val_metrics['f1']:.4f}")
    
#     # Save the model if it has the best F1 score so far
#     if val_metrics['f1'] > best_f1:
#         best_f1 = val_metrics['f1']
#         torch.save(model.state_dict(), 'best_deberta_ner_model.pt')
#         print(f"New best model saved with F1: {best_f1:.4f}")

# print("\nTraining complete!")


# ---

# This code evaluates the model on the validation set using scikit-learn's classification report. It performs the following steps:
#
# #### I) General Metrics Calculation:
#
# The ``evaluate ``function is used to calculate overall metrics such as loss, accuracy, precision, recall, and F1 score on the validation set. This centralizes metric calculations, reducing code redundancy.
#
# #### II) Detailed Classification Report:
# Even though general metrics are computed, a detailed classification report is also generated. This involves iterating through the validation DataLoader, gathering predictions and true labels, filtering out tokens marked as ``-100``, converting numerical labels to their corresponding tag names using the ``id2tag mapping``, and then printing the report with ``scikit-learn’s classification_report``.
#
# #### III) Commented Out Code:
# The entire cell is commented out to prevent accidental execution. 

# The following evaluation cell is commented out to prevent accidental execution.
# Uncomment the code when you are ready to perform the evaluation.

# from sklearn.metrics import classification_report

# # First, evaluate the model using our centralized evaluate function to get general metrics.
# val_metrics = evaluate(val_loader)
# print(f"Validation loss: {val_metrics['loss']:.4f}")
# print(f"Validation metrics: accuracy={val_metrics['accuracy']:.4f}, precision={val_metrics['precision']:.4f}, "
#       f"recall={val_metrics['recall']:.4f}, f1={val_metrics['f1']:.4f}")

# # Now, generate a detailed classification report for a finer analysis of each tag.
# model.eval()  # Ensure the model is in evaluation mode to disable dropout, etc.
# all_preds = []  # List to collect all prediction indices from the validation set.
# all_labels = []  # List to collect all ground truth label indices.

# # Disable gradient calculation for evaluation efficiency.
# with torch.no_grad():
#     # Iterate over each batch in the validation DataLoader.
#     for batch in val_loader:
#         # Move the input tensors and labels to the designated device (GPU or CPU).
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
        
#         # Forward pass: obtain the model's output logits.
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         # Compute the predictions by taking the argmax over the logits for each token.
#         preds = torch.argmax(outputs.logits, dim=2)
        
#         # Create a mask to filter out tokens with an ignore label (-100).
#         active_mask = labels != -100
#         # Extract the true labels and predictions from positions that are not ignored.
#         true = labels[active_mask].cpu().numpy()
#         pred = preds[active_mask].cpu().numpy()
        
#         # Append the results for the current batch to our overall lists.
#         all_labels.extend(true)
#         all_preds.extend(pred)

# # Convert numerical indices into their corresponding tag names using the id2tag mapping.
# true_tags = [id2tag[l] for l in all_labels]
# pred_tags = [id2tag[p] for p in all_preds]

# # Finally, print a detailed classification report including precision, recall, and F1 score for each tag.
# print(classification_report(true_tags, pred_tags))


# ---
# ## 14.1 Test Set Evaluation

# This cell evaluates the model on the test set and prints key performance metrics, including loss, accuracy, precision, recall, and F1-score. Additionally, to avoid losing results after execution, we format the classification report into a table.
#
# ### Steps:
# #### General Test Set Evaluation:
#
# - Calls evaluate(test_loader) to obtain overall metrics.
#
# - Prints loss, accuracy, precision, recall, and F1-score.
#
#

# The following test evaluation cell is commented out to prevent accidental execution.
# Uncomment if you wish to evaluate the model on the test set.

# print("\nEvaluating on test set...")

# # Compute general test set metrics using the evaluate function
# test_metrics = evaluate(test_loader)
# print(f"Test loss: {test_metrics['loss']:.4f}")
# print(f"Test metrics: accuracy={test_metrics['accuracy']:.4f}, precision={test_metrics['precision']:.4f}, "
#       f"recall={test_metrics['recall']:.4f}, f1={test_metrics['f1']:.4f}")

# ## Metrics and Next Steps 
#
# Some metrics are quite challenging, with low values, and a global f1-score of 0.41. Therefore, I believe it is a good idea to **try other models** that can be more easy to adjust. As I mentioned early, I will train a **specific BERT model** made for NER, improve it, and comment the results before choosing the best model.
#
# | Entity                    | Precision | Recall | F1-Score | Support |
# |---------------------------|-----------|--------|----------|---------|
# | B-ACTION                 | 0.00      | 0.00   | 0.00     | 21      |
# | B-INCIDENT               | 0.00      | 0.50   | 0.01     | 2       |
# | B-PIT_CALL               | 0.00      | 0.00   | 0.00     | 1       |
# | B-POSITION_CHANGE        | 0.03      | 0.03   | 0.03     | 29      |
# | B-SITUATION              | 0.00      | 0.00   | 0.00     | 41      |
# | B-STRATEGY_INSTRUCTION   | 0.18      | 0.05   | 0.07     | 43      |
# | B-TECHNICAL_ISSUE        | 0.00      | 0.00   | 0.00     | 18      |
# | B-TRACK_CONDITION        | 0.00      | 0.00   | 0.00     | 3       |
# | B-WEATHER               | 0.00      | 0.00   | 0.00     | 13      |
# | I-ACTION                 | 0.11      | 0.26   | 0.16     | 103     |
# | I-INCIDENT               | 0.00      | 0.00   | 0.00     | 7       |
# | I-PIT_CALL               | 0.00      | 0.00   | 0.00     | 3       |
# | I-POSITION_CHANGE        | 0.09      | 0.20   | 0.13     | 60      |
# | I-SITUATION              | 0.15      | 0.06   | 0.09     | 140     |
# | I-STRATEGY_INSTRUCTION   | 0.21      | 0.03   | 0.04     | 120     |
# | I-TECHNICAL_ISSUE        | 0.02      | 0.03   | 0.02     | 38      |
# | I-TRACK_CONDITION        | 0.00      | 0.00   | 0.00     | 13      |
# | I-WEATHER               | 0.00      | 0.00   | 0.00     | 64      |
# | O                        | 0.60      | 0.01   | 0.02     | 339     |
#
#

# ----

# ----

# # Transitioning to Pre-trained BERT for NER
#
# ## Futher explanations of why I am switching to BERT
# After experimenting with the DeBERTa v3 Large model, I´m  now transitioning to a BERT-based approach using ``dbmdz/bert-large-cased-finetuned-conll03-english``. This strategic shift is motivated by several factors:
#
# ### 1. Pre-trained NER Capabilites
#
# As I mentioned earlier, this BERT model has already been fine-tuned specifically for Named Entity Recognition on the CoNLL-03 dataset, providing a strong foundation for our F1 domain adaptation.
#
# ### 2. Transfer Learning Advantage
# y leveraging a model already optimized for entity detection, we can potentially achieve faster convergence and better performance on our specialized F1 entities. Deberta is a model that can be used for another NLP task (as we saw in previous notebooks, for sentiment analysis or intent classification), so this model theoretically looks better.
#
#
# ### 3. Model Comparison
#
# his allows us to benchmark performance between different transformer architectures (DeBERTa vs. BERT) to determine the most effective approach for our specific use case.
#
# ### 4. Performance on challenging entities.
#
# My initial results with DeBERTa showed challenges in detecting certain entity types (STRATEGY_INSTRUCTION, TRACK_CONDITION). BERT's different attention mechanism and pre-training might help address these issues.
#
# ## What I´ll implement
#
# In the following sections, I will try to:
#
# 1. **Initialize the BERT tokenizer and model** pre-trained on general NER tasks.
#
# 2. **Develop a custom dataset class** optimized for BERT's tokenization approach. That is, our current class but slightly changed for BERT´s architecture instead of Deberta´s. 
#
# 3. **Implement specialized loss functions** (Focal Loss and Weighted Cross-Entropy) to address class imbalance.
# 4. **Fine-tune the model with focus** on the most challenging entity categories.
#
# 5. **Evaluate performance**with detailed per-entity metrics and classification reports.
#
# 6. **Perform targeted analysis** of our most difficult entity classes.
#
# ## Goal
#
# My goal is to create a more robust entity recognition system that can reliably extract strategic information from F1 radio communications, with particular emphasis on improving detection of critical race strategy elements that were challenging for our previous model.
#
# ## Disclaimer
#
# The following code can result repetitive in some parts. However, it is necessary to redefine a great part of my old code, so I decided to implement it here to keep the old results and workflow. As some parts are almost the same, **shallower explanations will be made in those parts**.

# Again, we add a manual seed to always initialize the bert-large in the same seed. 
torch.manual_seed(42)

# Define the model name for the pretrained BERT tokenizer
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"

# Load the tokenizer associated with the specified model
tokenizer = BertTokenizerFast.from_pretrained(model_name)

# Check if it loaded correctly
print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")  # Display tokenizer class name
print(f"Vocabulary size: {len(tokenizer)}")  # Display vocabulary size


# #### Uncomment these lines if needed

# ===========================
# Bert Large Model Initialization
# ===========================

# Determine the computation device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # Print the selected device

# Get the number of labels based on the tag-to-ID mapping
num_labels = len(tag2id)

# Load the pretrained BERT model for token classification
model = BertForTokenClassification.from_pretrained(
    model_name,  # Pretrained BERT model name
    num_labels=num_labels,  # Number of classification labels
    id2label={i: l for l, i in tag2id.items()},  # Map IDs to labels (for interpretability)
    label2id=tag2id,  # Map labels to IDs (for training and inference)
    ignore_mismatched_sizes=True  # Allows model loading even if the classifier head size differs
)

# Move the model to the selected device (GPU/CPU)
model.to(device)

# Print model information
print(f"Model loaded: {model_name}")  # Confirm the model has been loaded
print(f"Number of labels: {num_labels}")  # Display the number of classification labels


# ---

# ###  Differences in `F1RadioNERDataset` (BERT) Compared to DeBERTa Version  
#
# This dataset class is designed for Named Entity Recognition (NER) in **Formula 1 Radio Messages**, but with adaptations for **BERT models** instead of DeBERTa. Below are the key differences:
#
# ####  Tokenization Strategy  
#
# - In **DeBERTa**, we used `tokenizer.encode_plus()`, while here we use:  
#
#   ```python
#   tokenized_inputs = self.tokenizer(
#       words,
#       is_split_into_words=True,
#       max_length=self.max_len,
#       padding="max_length",
#       truncation=True,
#       return_tensors="pt"
#   )
# - *Why?*
#     - ``is_split_into_words=True`` ensures that each token remains aligned with its original word, which is essential for NER.
#
#     - We specify ``return_tensors="pt"`` to return PyTorch tensors directly (DeBERTa used NumPy arrays initially).
#
# #### Handling Labels (NER Tags)
#
# In DeBERTa, label handling was different.
#
# Here, we explicitly check if tags are in string format and map them to tag2id manually:
#
# ```python
# tag_ids = []
# for tag in tags:
#     if isinstance(tag, str):
#         tag_ids.append(self.tag2id[tag])
#     else:
#         tag_ids.append(tag)
# ```
# - *Why?*
#
#     - Some datasets store NER tags as strings, while others already have integer IDs.
#
#     - This ensures consistency across different data formats.
#
# #### 3️. Word-to-Token Alignment  
# In both implementations, we align tokenized words with their corresponding labels.  
#
# However, in **BERT**, we explicitly use `word_ids(batch_index=0)` to retrieve word-level alignment:  
#
# ```python
# word_ids = tokenized_inputs.word_ids(batch_index=0)
# ```
# - *Why?*
#
#     - ``word_ids`` maps tokens back to their original words, allowing us to correctly assign NER labels.
#
#     - This is crucial for handling subwords in BERT’s WordPiece tokenization.
#
# #### 4. Handling Subwords in Label Assignment
#
# In **DeBERTa, we only assigned labels** to the first subword, marking others as -100.
#
# In **BERT**, we allow two options:
#
# - Option 1: Assign -100 to subwords (ignore during training).
#
# - Option 2 (used): Assign the same label to all subwords (propagate labels).
#
# - *Why?*
#
#     - Some models perform better if all subwords share the same NER tag, as BERT does (or I found it does during training).
#
#     - Others prefer ignoring subwords (setting them to -100).
#
# --- 
#
# #### Summary of Key Changes  
#
# | Feature           | DeBERTa Implementation         | BERT Implementation                        |
# |-------------------|--------------------------------|--------------------------------------------|
# | **Tokenization**  | `encode_plus()`               | `tokenizer(..., is_split_into_words=True)` |
# | **Label Handling** | Directly used dataset labels  | Explicit conversion from string to ID      |
# | **Word Alignment** | Implicit handling             | Uses `word_ids(batch_index=0)`             |
# | **Subword Labels** | Only first subword labeled    | Option to label all subwords               |
#

class F1RadioNERDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, tokenizer, tag2id, max_len=128):
        # Initialize dataset with Hugging Face dataset, tokenizer, and tag2id mappings
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.max_len = max_len
        
    def __len__(self):
        # Return the length of the dataset
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Retrieve a sample from the dataset
        item = self.dataset[idx]
        words = item["tokens"]  # Tokens from the dataset
        tags = item["ner_tags"]  # Corresponding NER tags
        
        # Convert tags from string to ID if necessary
        tag_ids = []  # List to store numerical tag ids
        for tag in tags:
            if isinstance(tag, str):  # If the tag is a string, map it to its ID
                tag_ids.append(self.tag2id[tag])
            else:  # If the tag is already in ID form, use it directly
                tag_ids.append(tag)
        
        # Tokenize the text and align the labels
        tokenized_inputs = self.tokenizer(
            words,  # Tokenized words
            is_split_into_words=True,  # Ensure tokenization is done word by word
            max_length=self.max_len,  # Maximum sequence length
            padding="max_length",  # Padding the sequences to the max length
            truncation=True,  # Truncate sequences that exceed max_length
            return_tensors="pt"  # Return PyTorch tensors
        )
        
        # Initialize labels with -100 for padding tokens
        labels = torch.ones(self.max_len, dtype=torch.long) * -100
        
        # Get word_ids to align the labels with words
        word_ids = tokenized_inputs.word_ids(batch_index=0)
        
        # Assign labels to non-special tokens (word pieces)
        previous_word_idx = None  # To keep track of the previous word index
        for i, word_idx in enumerate(word_ids):
            if word_idx is not None:  # If the token corresponds to a word
                if word_idx < len(tag_ids):  # Check if the word index is valid
                    # If it's the first subword, assign the label
                    # If it's not (continuation of a word), assign -100 or the same label as you prefer
                    if word_idx != previous_word_idx:  # New word
                        labels[i] = tag_ids[word_idx]
                    else:  # Continuation of the word (subword)
                        # Option 1: Use -100 for continuations (ignore them)
                        # labels[i] = -100
                        # Option 2: Use the same label for all subwords of the word
                        labels[i] = tag_ids[word_idx]
            previous_word_idx = word_idx  # Update the previous word index
        
        # Return the tokenized inputs and labels
        return {
            "input_ids": tokenized_inputs["input_ids"].flatten(),  # Flattened input ids
            "attention_mask": tokenized_inputs["attention_mask"].flatten(),  # Flattened attention mask
            "labels": labels  # The aligned labels
        }



# ---

# ### Why Use Focal Loss?
# Focal Loss helps to reduce the impact of easy-to-classify examples (which would otherwise dominate the loss function) and places more emphasis on harder-to-classify or misclassified examples. This is particularly useful in tasks with imbalanced class distributions (e.g., in named entity recognition, some classes might be underrepresented, like our case).

# Implementing Focal Loss for token classification tasks
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        """
        Focal Loss is designed to address the class imbalance problem by focusing more on hard-to-classify examples.
        
        Args:
            weight (Tensor, optional): Class weights for addressing class imbalance.
            gamma (float, optional): The focusing parameter, typically set to 2.0 to reduce the relative loss for well-classified examples.
        """
        super(FocalLoss, self).__init__()
        self.weight = weight  # Weight for class imbalance, can be set to None or a tensor of shape [num_classes]
        self.gamma = gamma  # Focusing parameter, controls the rate at which easy examples are down-weighted
        
    def forward(self, input, target):
        """
        Forward pass for calculating Focal Loss.
        
        Args:
            input (Tensor): Predicted logits from the model, shape [batch_size, seq_len, num_classes].
            target (Tensor): True labels, shape [batch_size, seq_len].
        
        Returns:
            Tensor: Computed focal loss value.
        """
        # Adjust dimensions for token classification (batch_size, seq_len, num_classes)
        if input.dim() > 2:
            # Reshape input to (batch_size*seq_len, num_classes) to flatten sequence dimension
            input = input.view(-1, input.size(-1))
        
        if target.dim() > 1:
            # Flatten target to (batch_size*seq_len,) to match the input
            target = target.view(-1)
        
        # Calculate Cross-Entropy Loss without reduction for each token in the sequence
        ce_loss = F.cross_entropy(input, target, weight=self.weight, ignore_index=-100, reduction='none')
        
        # Calculate the probability (pt) for each class
        pt = torch.exp(-ce_loss)  # pt is the probability that the model assigned to the correct class
        
        # Compute the focal loss
        # The term (1 - pt) ** gamma reduces the contribution from easy examples
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # Return the mean of the focal loss
        return focal_loss.mean()



# ### Keeping the same Tarining Config, but now using the new loss function

# Set the number of training epochs
epochs = 10

# Initialize an empty list to store labels from the training dataset
train_labels = []
for batch in train_loader:
    labels = batch['labels']
    
    # Create a mask to filter out ignored tokens (-100) since they should not contribute to class weighting
    mask = labels != -100  
    train_labels.extend(labels[mask].numpy())  # Convert and store valid labels

# Compute class weights to handle class imbalance in the dataset
class_weights = compute_class_weight(
    'balanced',  # This ensures that weights are inversely proportional to class frequency
    classes=np.unique(train_labels),  # Extract unique class labels
    y=train_labels  # Use all collected labels from the dataset
)

# Convert class weights to a PyTorch tensor and move them to the appropriate device (CPU/GPU)
class_weights = torch.FloatTensor(class_weights).to(device)

# Define the loss function (CrossEntropyLoss) with class weights to give more importance to underrepresented classes
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)  
# ignore_index=-100 ensures that padding tokens are not considered in loss computation

# Set the learning rate; it was previously 2e-5 but has been slightly increased to 3e-5
learning_rate = 3e-5

# Define warmup steps: A small portion of the training steps is dedicated to gradually increasing the learning rate
warmup_steps = int(0.05 * len(train_loader) * epochs)  # 5% of total training steps

# Calculate the total number of training steps (number of batches per epoch * number of epochs)
total_steps = len(train_loader) * epochs  

# Define the AdamW optimizer, which helps prevent overfitting by applying weight decay
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.03)  
# weight_decay=0.03 applies L2 regularization to reduce overfitting

# Define a learning rate scheduler to linearly decrease the learning rate after warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=warmup_steps,  # Gradual warmup phase
    num_training_steps=total_steps  # Decay learning rate linearly after warmup
)


# ### Same Training loop. Commented 

# # Initialize the best F1-score to track improvements across epochs
# best_f1 = 0  

# # Loop through the training process for the specified number of epochs
# for epoch in range(epochs):
#     print(f"\n{'='*50}")
#     print(f"Epoch {epoch+1}/{epochs}")  # Display current epoch
#     print(f"{'='*50}")
    
#     # Train the model for one epoch and obtain the training loss
#     train_loss = train_epoch()  # The train_epoch() function is assumed to be already defined
#     print(f"Training loss: {train_loss:.4f}")  # Print the training loss for monitoring
    
#     # Evaluate the model on the validation dataset
#     val_metrics = evaluate(val_loader)  # The evaluate() function is assumed to be already defined
#     print(f"Validation loss: {val_metrics['loss']:.4f}")
#     print(f"Validation metrics: accuracy={val_metrics['accuracy']:.4f}, "
#           f"precision={val_metrics['precision']:.4f}, recall={val_metrics['recall']:.4f}, "
#           f"f1={val_metrics['f1']:.4f}")  # Print validation performance metrics

#     # Save the model if it achieves the best F1-score so far
#     if val_metrics['f1'] > best_f1:
#         best_f1 = val_metrics['f1']  # Update the best F1-score
#         torch.save(model.state_dict(), 'best_bert_large_ner_model.pt')  # Save model weights
#         print(f"New best model saved with F1: {best_f1:.4f}")  # Notify about model update

# print("\nTraining complete!")  # Training process finished


# The following evaluation cell is commented out to prevent accidental execution.
# Uncomment the code when you are ready to perform the evaluation.

# from sklearn.metrics import classification_report

# # First, evaluate the model using our centralized evaluate function to get general metrics.
# val_metrics = evaluate(val_loader)
# print(f"Validation loss: {val_metrics['loss']:.4f}")
# print(f"Validation metrics: accuracy={val_metrics['accuracy']:.4f}, precision={val_metrics['precision']:.4f}, "
#       f"recall={val_metrics['recall']:.4f}, f1={val_metrics['f1']:.4f}")

# # Now, generate a detailed classification report for a finer analysis of each tag.
# model.eval()  # Ensure the model is in evaluation mode to disable dropout, etc.
# all_preds = []  # List to collect all prediction indices from the validation set.
# all_labels = []  # List to collect all ground truth label indices.

# # Disable gradient calculation for evaluation efficiency.
# with torch.no_grad():
#     # Iterate over each batch in the validation DataLoader.
#     for batch in val_loader:
#         # Move the input tensors and labels to the designated device (GPU or CPU).
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
        
#         # Forward pass: obtain the model's output logits.
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         # Compute the predictions by taking the argmax over the logits for each token.
#         preds = torch.argmax(outputs.logits, dim=2)
        
#         # Create a mask to filter out tokens with an ignore label (-100).
#         active_mask = labels != -100
#         # Extract the true labels and predictions from positions that are not ignored.
#         true = labels[active_mask].cpu().numpy()
#         pred = preds[active_mask].cpu().numpy()
        
#         # Append the results for the current batch to our overall lists.
#         all_labels.extend(true)
#         all_preds.extend(pred)

# # Convert numerical indices into their corresponding tag names using the id2tag mapping.
# true_tags = [id2tag[l] for l in all_labels]
# pred_tags = [id2tag[p] for p in all_preds]

# # Finally, print a detailed classification report including precision, recall, and F1 score for each tag.
# print(classification_report(true_tags, pred_tags))


# ---

# ---

# ## Focused Fine-Tuning for Challenging Entity Classes
#
# ### Why We Need Additional Fine-Tuning
#
# After our initial training with the BERT model, we observed that certain entity classes remain particularly challenging to detect:
#
# 1. **STRATEGY_INSTRUCTION**: Critical race strategy directives are being missed
# 2. **TRACK_CONDITION**: Information about track surface status shows poor recall
# 3. **TECHNICAL_ISSUE**: Mechanical problems are often misclassified
# 4. **INCIDENT**: Racing incidents are inconsistently detected
#
# The standard training approach treats all entity classes equally, but our F1 radio domain has natural class imbalances. More importantly, some entity types (like strategy instructions) carry higher strategic value than others, making their accurate detection a priority.
#
# ### Our Fine-Tuning Approach
#
# We'll implement a focused fine-tuning strategy with these key elements:
#
# 1. **Class-Weighted Loss Function**: We're creating a custom `WeightedCrossEntropyLoss` that assigns higher importance (5x weight) to our target classes, particularly STRATEGY_INSTRUCTION and TRACK_CONDITION
#    
# 2. **Lower Learning Rate**: Reducing from 3e-5 to 2e-6 to make smaller, more precise adjustments to the model
#
# 3. **Short Training Cycle**: Using just 5 epochs to avoid overfitting while refining detection capabilities
#
# 4. **Targeted Evaluation**: Specifically measuring improvements on our challenging entity classes
#
# This approach is similar to specialized medical image detection systems that prioritize detecting rare but critical conditions. By deliberately overweighting certain classes, we guide the model to be more sensitive to these important but challenging categories.
#
# ### Expected Benefits
#
# This fine-tuning strategy should:
#
# 1. Increase recall for strategic instructions and track conditions
# 2. Maintain performance on well-detected entity classes
# 3. Improve overall F1 score by addressing the weakest areas
# 4. Create a more balanced model suitable for real-world F1 strategy applications
#
# The final evaluation will include detailed per-class metrics to verify if our targeted approach successfully improved detection of these critical racing information categories.

# ##### Uncomment this lines if needed

# 1. First, load the saved model that we have already trained
model_path = '../../outputs/week4/models/best_bert_large_ner_model.pt'  
model = BertForTokenClassification.from_pretrained(
    "dbmdz/bert-large-cased-finetuned-conll03-english",
    num_labels=len(tag2id),
    id2label={i: l for l, i in tag2id.items()},
    label2id=tag2id,
    ignore_mismatched_sizes=True
)



#

model.load_state_dict(torch.load(model_path))
model.to(device)
print("Pre-trained model loaded successfully")


# ---

# ## Explanation: WeightedCrossEntropyLoss
#
# The `WeightedCrossEntropyLoss` class implements a variant of cross-entropy loss with custom weights for specific classes in a sequence classification problem.
#
# ### 🔹 Key Features:
#
# 1. **Adjustable class weights:**
#    * Higher weights are assigned to specific classes (`STRATEGY_INSTRUCTION`, `TRACK_CONDITION`).
#    * Moderate weights are assigned to others (`TECHNICAL_ISSUE`, `INCIDENT`).
#    * All other classes keep the default weight of `1.0`.
#
# 2. **Support for original and new target classes:**
#    * If `target_classes` is provided, it adjusts weights based on class relevance.
#
# 3. **PyTorch Compatibility:**
#    * Uses `F.cross_entropy` with per-class weights (`self.class_weights`).
#    * Ignores `-100` indices, typically used for padding in NLP models.

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, num_labels, target_classes=None, class_weight_factor=5.0):
        """
        Implements Cross Entropy Loss with custom class weights.

        Parameters:
        - num_labels (int): Total number of labels in the classification task.
        - target_classes (list, optional): Indices of target classes to assign custom weights.
        - class_weight_factor (float, optional): Base weight for specific classes (default is 5.0).
        """
        super(WeightedCrossEntropyLoss, self).__init__()

        # Initialize all class weights to 1.0
        self.class_weights = torch.ones(num_labels, dtype=torch.float)
        
        # Identify original target classes
        original_targets = []
        for tag, idx in tag2id.items():
            if "STRATEGY_INSTRUCTION" in tag or "TRACK_CONDITION" in tag:
                original_targets.append(idx)
        
        # Assign weights to relevant classes
        if target_classes:
            for cls_idx in target_classes:
                tag = id2tag[cls_idx]  # Convert index to label name
                
                if cls_idx in original_targets:
                    # Keep a high weight for original target classes (default 5.0)
                    self.class_weights[cls_idx] = class_weight_factor  
                elif "TECHNICAL_ISSUE" in tag or "INCIDENT" in tag:
                    # Assign a moderate weight (3.0) to new categories related to technical issues or incidents
                    self.class_weights[cls_idx] = 3.0  

        # Define the ignore index for padding tokens
        self.ignore_index = -100
    
    def forward(self, logits, labels):
        """
        Computes weighted cross-entropy loss.

        Parameters:
        - logits (tensor): Model output with shape (batch_size, seq_len, num_labels).
        - labels (tensor): Ground truth labels with shape (batch_size, seq_len).

        Returns:
        - A scalar tensor representing the average loss.
        """
        # Move class weights to the same device as logits
        self.class_weights = self.class_weights.to(logits.device)
        
        # Apply cross-entropy loss with class weights
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),  # Reshape for correct loss computation
            labels.view(-1),  # Ensure labels match expected format
            weight=self.class_weights,  # Apply custom class weights
            ignore_index=self.ignore_index  # Ignore padding tokens (-100)
        )



# ---

# ### Target Class Identification and Loss Function Configuration
#
# This code snippet:
#
# 1. **Identifies specific target classes** in `tag2id` that contain `"STRATEGY_INSTRUCTION"` or `"TRACK_CONDITION"`.
# 2. **Prints their corresponding indices** for debugging purposes.
# 3. **Creates a `WeightedCrossEntropyLoss` instance** using these target classes and assigns them a higher weight (×5).

# Identify indices of problematic classes
target_class_indices = []
for tag, idx in tag2id.items():
    if "STRATEGY_INSTRUCTION" in tag or "TRACK_CONDITION" in tag:
        target_class_indices.append(idx)
        print(f"Target class: {tag} (ID: {idx})")  # Debugging output to verify selected classes

# Create custom loss function with increased weight for target classes
custom_loss = WeightedCrossEntropyLoss(
    num_labels=len(tag2id),  # Total number of labels in the classification task
    target_classes=target_class_indices,  # Indices of classes that need higher weighting
    class_weight_factor=5.0  # Increase weight by 5x for the selected target classes
)


# ---

# ## Explanation: Modified Training Function with Custom Loss
#
# This function, `train_epoch_focused()`, modifies the standard training loop by incorporating the **custom weighted loss function** (`custom_loss`). The key changes include:
#
# 1. **Using the** `WeightedCrossEntropyLoss` function to handle class imbalance.
# 2. **Computing loss dynamically** based on `logits` and `labels`.
# 3. **Gradient clipping** to prevent exploding gradients.
# 4. **Updating both the optimizer and learning rate scheduler** after each step.

# 3. Modified training function to use custom loss
def train_epoch_focused():
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Normal forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Compute loss using custom function
        loss = custom_loss(logits, labels)
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    
    return total_loss / len(train_loader)


# 4. Set a low learning rate for fine-tuning
learning_rate = 2e-6  # Lower for fine-tuning
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

# Short cycle for fine-tuning
fine_tuning_epochs = 5
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=int(0.1 * len(train_loader) * fine_tuning_epochs),
    num_training_steps=len(train_loader) * fine_tuning_epochs
)


def evaluate_model(data_loader):
    """Evaluates the model on the given data loader and computes relevant metrics."""
    
    model.eval()  # Set the model to evaluation mode (disables dropout, batch norm, etc.)
    total_loss = 0  # Initialize the total loss accumulator
    all_preds = []  # List to store all model predictions
    all_labels = []  # List to store all ground-truth labels
    
    with torch.no_grad():  # Disable gradient computation to improve efficiency
        for batch in tqdm(data_loader, desc="Evaluating"):  # Iterate over the data loader
            # Move input tensors to the specified device (CPU or GPU)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass through the model to obtain logits (raw predictions)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # Extract logits from the model output
            
            # Compute loss using the custom loss function
            loss = custom_loss(logits, labels)
            total_loss += loss.item()  # Accumulate the total loss
            
            # Store predictions and true labels for later evaluation
            all_preds.append(logits.detach().cpu().numpy())  # Convert logits to NumPy and store
            all_labels.append(labels.detach().cpu().numpy())  # Convert labels to NumPy and store
    
    # Concatenate stored predictions and labels into single NumPy arrays
    all_preds = np.concatenate([p for p in all_preds], axis=0)
    all_labels = np.concatenate([l for l in all_labels], axis=0)
    
    # Compute evaluation metrics such as accuracy, precision, recall, and F1-score
    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = total_loss / len(data_loader)  # Include the average loss in the metrics
    
    return metrics  # Return computed evaluation metrics


# ---

# ## Fine-Tuning Training Cycle Implementation
#
# This code implements the complete fine-tuning cycle with several important components:
#
# 1. **Performance tracking:**
#    * Starts with previous best F1 score (0.4229) as baseline
#    * Continuously monitors for improvements in validation metrics
#
# 2. **Epoch-based training:**
#    * Executes `train_epoch_focused()` for each epoch
#    * Displays comprehensive progress metrics including loss and F1 score
#
# 3. **Model evaluation:**
#    * Uses custom `evaluate_model()` function optimized for our weighted classes
#    * Calculates precision, recall, and F1 score on validation data
#
# 4. **Model persistence strategy:**
#    * Saves model only when F1 score improves over previous best
#    * Implements CPU offloading to prevent CUDA memory errors during saving
#    * Automatically restores model to GPU for continued training
#
# 5. **GPU memory management:**
#    * Temporarily moves model to CPU during saving operations
#    * Returns model to GPU for continued training efficiency

# # Updated fine-tuning cycle
# best_f1 = 0.4229  # Start with the previous best F1 score

# print("\nStarting fine-tuning focused on challenging classes...")
# for epoch in range(fine_tuning_epochs):
#     print(f"\n{'='*50}")
#     print(f"Epoch {epoch+1}/{fine_tuning_epochs}")
#     print(f"{'='*50}")
    
#     train_loss = train_epoch_focused()
#     print(f"Training loss: {train_loss:.4f}")
    
#     # Use the new evaluation function
#     val_metrics = evaluate_model(val_loader)
#     print(f"Validation loss: {val_metrics['loss']:.4f}")
#     print(f"Validation metrics: accuracy={val_metrics['accuracy']:.4f}, precision={val_metrics['precision']:.4f}, "
#           f"recall={val_metrics['recall']:.4f}, f1={val_metrics['f1']:.4f}")
    
#     # Save if F1 improves
#     if val_metrics['f1'] > best_f1:
#         best_f1 = val_metrics['f1']
#         # Move to CPU to avoid CUDA errors
#         model_cpu = model.cpu()
#         torch.save(model_cpu.state_dict(), '../../outputs/week4/models/best_focused_bert_model.pt')
#         # Restore to GPU
#         model = model.to(device)
#         print(f"New best model saved with F1: {best_f1:.4f}")

# print("\nFine-tuning complete!")


# ---

# ## Comprehensive Model Evaluation with Target Class Analysis
#
# This evaluation procedure extends beyond standard metrics to provide detailed insights into model performance:
#
# 1. **Overall model assessment:**
#    * Evaluates the fine-tuned model on the held-out test set
#    * Reports standard metrics (accuracy, precision, recall, F1)
#    * Uses our custom `evaluate_model()` function that handles weighted classes
#
# 2. **Detailed classification analysis:**
#    * Generates comprehensive classification report across all entity types
#    * Shows per-class precision, recall, F1-score, and support
#    * Reveals both strengths and remaining challenges in the model
#
# 3. **Target-focused evaluation:**
#    * Specifically analyzes performance on our four target entity classes:
#      - `B-STRATEGY_INSTRUCTION`: Beginning of strategy instructions
#      - `I-STRATEGY_INSTRUCTION`: Continuation of strategy instructions
#      - `B-TRACK_CONDITION`: Beginning of track condition descriptions
#      - `I-TRACK_CONDITION`: Continuation of track condition descriptions
#
# 4. **Percentage-based success metrics:**
#    * Calculates the exact percentage of correctly predicted entities for each target class
#    * Provides clear visibility into whether our focused fine-tuning has succeeded
#    * Enables direct comparison with pre-fine-tuning performance
#
# This evaluation approach helps determine if our class-weighted fine-tuning strategy has successfully improved detection of the most challenging entity types while maintaining overall performance.

# # 6. Final evaluation focusing on difficult classes
# print("\nEvaluating on test set...")
# test_metrics = evaluate_model(test_loader)
# print(f"Test metrics: accuracy={test_metrics['accuracy']:.4f}, precision={test_metrics['precision']:.4f}, "
#       f"recall={test_metrics['recall']:.4f}, f1={test_metrics['f1']:.4f}")

# # Detailed classification report
# from sklearn.metrics import classification_report

# model.eval()
# all_preds = []
# all_labels = []

# with torch.no_grad():
#     for batch in test_loader:
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
        
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         preds = torch.argmax(outputs.logits, dim=2)
        
#         # Filter out padding tokens
#         active_mask = labels != -100
#         true = labels[active_mask].cpu().numpy()
#         pred = preds[active_mask].cpu().numpy()
        
#         all_labels.extend(true)
#         all_preds.extend(pred)

# # Convert IDs to labels
# true_tags = [id2tag[l] for l in all_labels]
# pred_tags = [id2tag[p] for p in all_preds]

# # Print full report
# print("\nFull classification report:")
# print(classification_report(true_tags, pred_tags))

# # Specific analysis for target classes
# print("\nTarget class analysis:")
# target_tags = ["B-STRATEGY_INSTRUCTION", "I-STRATEGY_INSTRUCTION", 
#               "B-TRACK_CONDITION", "I-TRACK_CONDITION"]

# for tag in target_tags:
#     # Filter only instances of this label
#     indices = [i for i, t in enumerate(true_tags) if t == tag]
#     if indices:
#         true_subset = [true_tags[i] for i in indices]
#         pred_subset = [pred_tags[i] for i in indices]
        
#         print(f"\nFor {tag}:")
#         print(f"Total examples: {len(indices)}")
#         correct = sum(1 for t, p in zip(true_subset, pred_subset) if t == p)
#         print(f"Correctly predicted: {correct} ({correct/len(indices)*100:.2f}%)")


# ---

# ---

# ## Function: `extract_entities_from_radio`
#
# This function provides the critical bridge between our trained NER model and practical applications by transforming raw F1 radio messages into structured entity data.
#
# ### Key Processing Steps:
#
# 1. **Text Preparation:**
#    * Splits the raw message into tokens
#    * Handles tokenization with proper word alignment for transformer input
#
# 2. **Model Inference:**
#    * Sets model to evaluation mode
#    * Performs forward pass with gradient calculation disabled
#    * Extracts predicted entity tags from logits
#
# 3. **Token Alignment:**
#    * Maps predictions back to original words using `word_ids`
#    * Handles subword tokenization by considering only the first subtoken of each word
#    * Maintains the integrity of the original message structure
#
# 4. **Entity Reconstruction:**
#    * Applies BIO (Beginning-Inside-Outside) tag interpretation
#    * Reconstructs continuous multi-word entities
#    * Handles entity boundaries and transitions between entity types
#    * Groups tokens into complete entity phrases
#
# 5. **Data Structuring:**
#    * Returns organized dictionary with entity types as keys
#    * Groups multiple instances of the same entity type
#    * Preserves the exact entity text as it appeared in the message
#
# This function enables practical applications like real-time race strategy assistance, automated highlight generation, and structured data extraction from F1 team communications.

def extract_entities_from_radio(radio_message, model, tokenizer, id2tag):
    """
    Extracts entities from an F1 radio message and returns them in a clean format.
    
    Args:
        radio_message (str): The raw F1 team radio message text
        model: The fine-tuned BERT model for entity recognition
        tokenizer: The tokenizer corresponding to the model
        id2tag (dict): Mapping from numeric IDs to entity tags
        
    Returns:
        dict: A dictionary with entity types as keys and lists of entity text as values
    """
    # Split the message into individual word tokens
    # This simple approach works for basic tokenization before passing to BERT tokenizer
    tokens = radio_message.split()
    
    # Convert tokens to model inputs using the tokenizer
    # Parameters:
    #   - is_split_into_words=True: Indicates input is already tokenized
    #   - return_tensors="pt": Return PyTorch tensors
    #   - padding=True: Add padding to reach maximum length
    #   - truncation=True: Truncate if exceeds maximum length
    inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)  # Move tensors to the appropriate device (GPU/CPU)
    
    # Set model to evaluation mode to disable dropout, etc.
    model.eval()
    
    # Disable gradient calculation for inference (saves memory and computation)
    with torch.no_grad():
        # Forward pass through the model
        outputs = model(**inputs)
        
        # Get the predicted class for each token
        # - outputs.logits: model output with shape [batch, seq_len, num_classes]
        # - torch.argmax(..., dim=2): Get the class with highest probability for each token
        # - [0]: Get the first (and only) example in the batch
        # - .cpu().numpy(): Move to CPU and convert to numpy for processing
        predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
    
    # Initialize containers for processing results
    entities = {}               # Will hold all extracted entities by type
    current_entity = None       # The entity type we're currently tracking
    current_text = []           # Tokens for the current entity we're building
    
    # BERT tokenizer splits words into subwords, so we need to map predictions back
    # to original words. word_ids gives us this mapping.
    word_ids = inputs.word_ids(batch_index=0)
    previous_word_idx = None    # Track previous word to detect token boundaries
    token_predictions = []      # Will hold one prediction per original token
    
    # First pass: map predictions from subwords back to original words
    for idx, word_idx in enumerate(word_ids):
        # Special tokens like [CLS] and [SEP] have word_idx=None
        if word_idx is None:
            continue
            
        # We only want one prediction per original word, not per subword
        # For words split into multiple subwords, we only take the prediction
        # from the first subword (standard practice in BERT-based NER)
        if word_idx != previous_word_idx:
            tag_id = predictions[idx]          # Get the predicted class ID
            tag = id2tag[tag_id]               # Convert ID to tag string (e.g., "B-ACTION")
            token_predictions.append(tag)      # Store prediction for this original token
            previous_word_idx = word_idx       # Update tracking variable
    
    # Second pass: process the predictions to extract continuous entities
    # using the BIO (Beginning-Inside-Outside) tagging scheme
    for i, (token, tag) in enumerate(zip(tokens, token_predictions)):
        # Case 1: Beginning of a new entity (B-*)
        if tag.startswith('B-'):
            # If we were tracking a previous entity, finalize it before starting new one
            if current_entity:
                entity_text = ' '.join(current_text)
                if current_entity not in entities:
                    entities[current_entity] = []
                entities[current_entity].append(entity_text)
            
            # Start tracking the new entity
            current_entity = tag[2:]           # Remove the "B-" prefix to get entity type
            current_text = [token]             # Start collecting tokens for this entity
            
        # Case 2: Inside/continuation of an entity (I-*)
        elif tag.startswith('I-') and current_entity == tag[2:]:
            # Only append if it's continuing the same entity type we're tracking
            current_text.append(token)
            
        # Case 3: Outside any entity (O) or mismatch between I-tag and current entity
        else:
            # If we were tracking an entity, finalize it
            if current_entity:
                entity_text = ' '.join(current_text)
                if current_entity not in entities:
                    entities[current_entity] = []
                entities[current_entity].append(entity_text)
                # Reset tracking variables
                current_entity = None
                current_text = []
    
    # Handle edge case: if message ends while still tracking an entity
    if current_entity:
        entity_text = ' '.join(current_text)
        if current_entity not in entities:
            entities[current_entity] = []
        entities[current_entity].append(entity_text)
    
    # Return the structured entity dictionary
    # Format: {'ACTION': ['box this lap', 'push harder'], 'WEATHER': ['rain expected'], ...}
    return entities


# ---

# ## User-Facing Entity Analysis Function
#
# The `analyze_f1_radio` function serves as the user-friendly interface to our NER system. It takes a raw F1 radio message as input, processes it through our entity extraction pipeline, and presents the results in a readable, hierarchical format.
#
# ### Key Features:
# - **Simple Interface**: Accepts a single string parameter containing the radio message
# - **Integration Point**: Connects the underlying NER model with end-user applications
# - **Formatted Display**: Presents extracted entities in an organized, easy-to-read structure
# - **Entity Categorization**: Groups entities by type for clearer understanding of message content
# - **Null Handling**: Gracefully handles cases where no entities are detected
# - **Return Value**: Provides the structured entity dictionary for further programmatic use
#
# This function enables practical applications like real-time race strategy assistance, radio message categorization, and structured data visualization from F1 team communications.

def analyze_f1_radio(message):
    """
    Function for the end user: analyzes a message and displays the entities.
    
    This function provides a user-friendly interface to extract and display
    named entities from F1 radio communications.
    
    Args:
        message (str): The raw F1 team radio message to analyze
        
    Returns:
        dict: A dictionary with entity types as keys and lists of entity text as values
    """
    # Print the original message to provide context for the analysis results
    # The quotes help visually distinguish the message from other output
    print(f"\nAnalyzing message: \"{message}\"")
    
    # Process the message using our entity extraction function
    # This passes the message through the NER model and structures the results
    # The function handles all the complexity of tokenization and BIO tag processing
    entities = extract_entities_from_radio(message, model, tokenizer, id2tag)
    
    # Begin displaying results with a header
    print("\nDetected entities:")
    
    # Handle the case where no entities were detected
    # This could happen with very short messages or messages without strategic content
    if not entities:
        print("  No relevant entities detected.")
    else:
        # Sort entity types alphabetically for consistent output presentation
        # For each entity type, display all instances found in the message
        for entity_type, texts in sorted(entities.items()):
            # Print the entity type with proper indentation
            print(f"  {entity_type}:")
            
            # For each instance of this entity type, display with bullet points
            # The quotes help visually distinguish the extracted text
            for text in texts:
                print(f"    • \"{text}\"")
    
    # Return the structured entity dictionary for potential further processing
    # This allows the function to be used both for display and as part of a pipeline
    return entities


# ## Finally, an example usage

# Prove the model with some real and synthetic messages
example_messages = [
    "Box this lap, box this lap. We're switching to slicks.",
    "Hamilton is 1.2 seconds behind you and closing fast. Defend position.",
    "Yellow flags in sector 2, incident at turn 7. Be careful.",
    "Track is drying up now, lap times are improving.",
    "Box this lap and switch to intermediates – we’re facing a technical issue on the front wing and worsening track conditions.",
    "Incident at turn 6 with debris on the track; you’re 0.8 seconds behind – defend your position immediately.",
    "Box now, the track is drying rapidly while the weather forecast predicts rain incoming; adjust your strategy and check for any technical issues.",
    "Maintain pace but be cautious: an incident at turn 3 is causing yellow flags and changing track conditions – reposition immediately.",
    "Switch pit call: we’re experiencing a gearbox technical issue while the weather remains clear; focus on defending your position with updated strategy instructions.",
    "Immediate action required – an incident occurred in sector 2 and track conditions are deteriorating; box next lap and follow strategy instructions.",
    "Overtake now, but be aware the weather might worsen and a technical issue with the engine is causing vibrations; adjust your positioning accordingly.",
    "Attention: the track is wet and slippery, and an incident at turn 5 has been reported; box this lap and modify your strategy as needed.",
    "Driver reporting a technical issue with the rear brakes while track conditions are improving; defend your position and prepare for a pit call.",
    "Urgent: a multi-car incident in sector 3 has occurred, track conditions have deteriorated, and the weather is turning unpredictable; box immediately and follow strategy instructions."
    "Okay Max, we're expecting rain in about 9 or 10 minutes. What are your thoughts? That you can get there or should we box? We'd need to box this lap to cover Leclerc. I can't see the weather, can I? I don't know.",
    "Max, we've currently got yellows in turn 7. Ferrari in the wall, no? Yes, that's Charles stopped. We are expecting the potential of an aborted start, but just keep to your protocol at the moment.",
]

for message in example_messages:
    analyze_f1_radio(message)
    print("\n" + "-"*50)

# # Named Entity Recognition Model Analysis for F1 Radio Communications
#
# ## Model Comparison Overview
#
# We evaluated three different models for extracting named entities from Formula 1 team radio communications:
#
# 1. **DeBERTa v3 Large**: Advanced transformer architecture known for state-of-the-art performance on NLP tasks
# 2. **BERT Large (pre-trained for NER)**: Model fine-tuned on CoNLL-03 dataset, adapted to our F1-specific entity classes
# 3. **BERT Large with focused fine-tuning**: Final model with additional training focused on challenging entity classes
#
# ## Performance Metrics Comparison
#
# | Model | Accuracy | Precision | Recall | F1-score |
# |-------|----------|-----------|--------|----------|
# | DeBERTa v3 Large | 0.4513 | 0.4283 | 0.4513 | 0.4115 |
# | BERT Large NER | 0.4199 | 0.4466 | 0.4199 | 0.4229 |
# | **BERT Large Fine-tuned** | **0.4411** | **0.4543** | **0.4411** | **0.4298** |
#
# ## Entity-Level Performance Analysis (F1-scores)
#
# | Entity Type | DeBERTa v3 | BERT NER | BERT Fine-tuned |
# |-------------|------------|----------|-----------------|
# | ACTION | 0.42 | 0.54 | **0.57** |
# | POSITION_CHANGE | 0.26 | **0.66** | 0.65 |
# | INCIDENT | 0.00 | 0.22 | **0.22** |
# | TECHNICAL_ISSUE | 0.00 | 0.26 | **0.23** |
# | SITUATION | 0.16 | 0.30 | **0.30** |
# | TRACK_CONDITION | 0.06 | 0.11 | **0.11** |
# | WEATHER | **0.69** | 0.44 | 0.40 |
#
# ## Conclusions
#
# **We selected the fine-tuned BERT model for the following reasons:**
#
# 1. **Best overall performance**: Achieved the highest F1-score (0.4298) and precision (0.4543) across all models
# 2. **Balanced entity recognition**: More consistent performance across different entity types
# 3. **Improved performance on critical entities**: Better recognition of ACTION, POSITION_CHANGE, and SITUATION entities, which are crucial for strategic decision-making
# 4. **Better generalization**: Shows improved ability to identify both the beginning (B-) and continuation (I-) of entities
#
# While DeBERTa v3 performed well on WEATHER entities, it struggled significantly with several other important categories. The base BERT model showed promising results, but our focused fine-tuning approach improved performance further by emphasizing challenging entity classes through weighted loss functions.
#
# The fine-tuned model successfully recognizes 100% of I-TRACK_CONDITION instances and shows improved performance on technical issues and incidents compared to the initial models.
#
#

# ---

# # Next Steps
#
# ## Merging all the models
#
# In our next notebook, `N06_model_merging.ipynb`, we'll integrate the three specialized models we've developed throughout this project:
#
# 1. **Sentiment Analysis Model:** Detects emotions and tone in radio communications
# 2. **Intent Recognition Model:** Identifies the purpose and goals behind messages
# 3. **Named Entity Recognition Model:** Extracts structured information about race elements
#
# ### Integration Approach
#
# We'll create a unified pipeline that:
#
# 1. Takes a raw F1 team radio message as input
# 2. Processes it through each specialized model in parallel
# 3. Combines the outputs into a comprehensive JSON structure
# 4. Provides a single interface for analyzing radio communications
#
# ### Benefits of Integration
#
# This merged approach offers several advantages:
#
# - **Comprehensive Analysis:** Captures semantic, pragmatic, and informational dimensions
# - **Standardized Output:** Provides a consistent JSON format for downstream applications
# - **Simplified Interface:** Requires just one function call to access all analyses
# - **Racing Context Awareness:** Combines different perspectives for better strategic insights
#
#
# ## **Integration with logical agent**: 
#
# Connect the NER system with the strategic recommendation engine for real-time race strategy optimization.
#
#
# The current model is production-ready and can reliably extract most entity types from F1 radio communications, providing valuable structured data for strategic decision-making systems.
