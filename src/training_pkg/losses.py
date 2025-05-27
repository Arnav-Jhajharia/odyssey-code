# training_pkg/losses.py (or define in your main training script)
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin: float = 0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1: torch.Tensor, output2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # output1 and output2 are the embeddings of the two sequences
        # label is 1 if similar, 0 if dissimilar
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = False)
        
        # For similar pairs (label=1), loss is euclidean_distance^2
        # For dissimilar pairs (label=0), loss is max(0, margin - euclidean_distance)^2
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2) +
            (label)     * torch.pow(euclidean_distance, 2)
        )
        return loss_contrastive
