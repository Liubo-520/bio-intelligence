"""
loss.py — Focal Loss for handling class imbalance in DTI prediction.

Standard BCELoss treats all samples equally, which biases the model
towards the majority (negative) class. Focal Loss down-weights easy
negatives and focuses training on hard positives, which is crucial
for datasets like Davis where only ~5% of pairs are positive.

Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss with logits input (numerically stable).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t = sigmoid(logit) for positive class, 1-sigmoid(logit) for negative.

    With gamma=2, the loss for well-classified examples (p_t > 0.9) is
    reduced by 100x compared to standard cross-entropy, letting the model
    focus on ambiguous/hard samples where it can still improve.

    Args:
        alpha: weighting factor for positive class. Set > 0.5 for
               imbalanced datasets with fewer positives.
        gamma: focusing parameter. gamma=0 recovers standard BCE.
               gamma=2 is the recommended default.
        pos_weight: optional additional positive class weight (like BCEWithLogitsLoss).
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0,
                 pos_weight: torch.Tensor = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 1) raw model output (before sigmoid)
            targets: (B, 1) binary labels (0 or 1)

        Returns:
            scalar focal loss
        """
        # Standard BCE with logits (numerically stable)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        # p_t: probability of correct class
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # focal modulating factor
        focal_weight = (1 - p_t) ** self.gamma

        # alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * focal_weight * bce

        return loss.mean()
