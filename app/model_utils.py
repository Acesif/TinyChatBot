from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from config import DEVICE, MODEL_NAME

def load_model():
    """
    Load the transformer model with appropriate configurations.
    """
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map=None,
        torch_dtype=torch.bfloat16
    ).to(DEVICE)
    return model

def load_tokenizer():
    """
    Load the tokenizer and configure padding token.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
