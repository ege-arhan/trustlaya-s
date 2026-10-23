import random
import numpy as np

def device():
    import torch
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
def seed_all(seed=42):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
def normalize(text):
    return text.replace("I", "ı").lower()
