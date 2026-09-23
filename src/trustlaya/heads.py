import torch.nn as nn
from .labels import TASKS, ACTIONS, SEVERITIES
class MultiTaskHeads(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.risks = nn.Linear(hidden, len(TASKS))
        self.severity = nn.Linear(hidden, len(SEVERITIES))
        self.action = nn.Linear(hidden, len(ACTIONS))
    def forward(self, x):
        return self.risks(x), self.severity(x), self.action(x)
