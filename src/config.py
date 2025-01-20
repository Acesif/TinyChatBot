import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "Qwen/Qwen1.5-0.5B-Chat"
