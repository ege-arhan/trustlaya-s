import torch
import torch.nn.functional as F

def loss(outputs, risks, severity, action, teacher=None, teacher_weight=0.05):
    r,s,a=outputs
    task=F.binary_cross_entropy_with_logits(r,risks)+0.3*F.cross_entropy(s,severity)+0.3*F.cross_entropy(a,action)
    if teacher is not None:
        # Teacher probabilities are weak targets, never ground truth.
        mask=torch.isfinite(teacher) & ((teacher>=0.5)==(risks>=0.5))
        if mask.any(): task=task+teacher_weight*F.binary_cross_entropy_with_logits(r[mask],teacher[mask].clamp(0.01,0.99))
    return task
