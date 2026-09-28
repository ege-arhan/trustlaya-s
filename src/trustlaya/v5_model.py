"""V5 research candidate: 42M Turkish BERT encoder + one ATTACK/BENIGN logit.

Not connected to the firewall, CLI or policy engine. V2 stays the default.
Mean pooling matches V2. Inputs are HEAD-truncated to `content_tokens`
(<= 510: BERT has 512 positions including [CLS] and [SEP]).
"""

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModel

MAX_CONTENT = 510


class V5Classifier(nn.Module):
    def __init__(self, backbone, pretrained=True):
        super().__init__()
        self.encoder = (AutoModel.from_pretrained(backbone) if pretrained
                        else AutoModel.from_config(AutoConfig.from_pretrained(backbone)))
        self.dropout = nn.Dropout(0.1)
        self.attack = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.attack(self.dropout(pooled)).squeeze(-1)

    def save(self, path):
        save_file({k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}, str(path))

    def load(self, path):
        self.load_state_dict(load_file(str(path)))


def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    """Binary focal loss (Lin et al. 2017); alpha weights the positive (ATTACK) class."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t) ** gamma * bce).mean()


def batch(token_lists, content_tokens, cls_id, sep_id):
    """HEAD-truncate to content_tokens, add [CLS]/[SEP], pad dynamically."""
    if not 1 <= content_tokens <= MAX_CONTENT:
        raise ValueError(f"content_tokens must be 1..{MAX_CONTENT}")
    seqs = [[cls_id, *tokens[:content_tokens], sep_id] for tokens in token_lists]
    width = max(len(s) for s in seqs)
    ids = torch.zeros(len(seqs), width, dtype=torch.long)
    mask = torch.zeros(len(seqs), width, dtype=torch.long)
    for i, seq in enumerate(seqs):
        ids[i, :len(seq)] = torch.tensor(seq)
        mask[i, :len(seq)] = 1
    return ids, mask


def source_balanced_weights(groups):
    """Per-item sampling weights: ATTACK and BENIGN each get half the mass, split
    equally across the sources that supply that label. `groups` = [(source, label)]."""
    per_label_sources = {}
    counts = {}
    for group in groups:
        counts[group] = counts.get(group, 0) + 1
        per_label_sources.setdefault(group[1], set()).add(group[0])
    labels = len(per_label_sources)
    return [1.0 / labels / len(per_label_sources[g[1]]) / counts[g] for g in groups]
