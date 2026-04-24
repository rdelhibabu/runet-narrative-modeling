# RuBERT Fine-Tuning Script (Masked Language Modeling)
# Architecture: DeepPavlov/rubert-base-cased
import torch
from transformers import BertForMaskedLM, BertTokenizer

def load_model():
    tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
    model = BertForMaskedLM.from_pretrained('DeepPavlov/rubert-base-cased')
    return tokenizer, model

# Training configurations as per Table 4:
# Batch Size: 64 (Effective 256)
# Optimizer: AdamW
# Learning Rate: 2e-5
# Epochs: 3
