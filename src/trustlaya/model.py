from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoModel
from safetensors.torch import load_file, save_file
from .heads import MultiTaskHeads

BACKBONE = str(Path(__file__).resolve().parents[2] / "models/base")
class TrustLaya(nn.Module):
    def __init__(self, backbone=BACKBONE, pretrained=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone) if pretrained else AutoModel.from_config(__import__('transformers').AutoConfig.from_pretrained(backbone))
        self.heads = MultiTaskHeads(self.encoder.config.hidden_size)
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        masked = out.last_hidden_state * attention_mask.unsqueeze(-1)
        pooled = masked.sum(1) / attention_mask.sum(1, keepdim=True).clamp(min=1)
        return self.heads(pooled)
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_file({k:v.detach().cpu().contiguous() for k,v in self.state_dict().items()}, str(path))
    def load(self, path):
        self.load_state_dict(load_file(str(path)))
