import os
from dotenv import load_dotenv
from huggingface_hub import login


def setup():
    """Load environment variables and login to Hugging Face."""
    load_dotenv()
    token = os.getenv("token")

    if not token:
        raise ValueError("HUGGINGFACE_TOKEN not found in environment variables!")

    login(token=token)
