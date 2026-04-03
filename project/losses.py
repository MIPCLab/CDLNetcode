import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDecouplingLoss(nn.Module):
    """
    类内特征解耦损失：惩罚同类样本特征向量之间的非对角相似度，
    鼓励同类样本在特征空间中保持多样性。
    """

    def __init__(self):
        super().__init__()

    def forward(self, features, labels):
        loss = 0.0
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            class_features = features[labels == label]
            if class_features.size(0) > 1:
                class_features = F.normalize(class_features, p=2, dim=-1)
                sim = torch.matmul(class_features, class_features.t())
                off_diag = sim - torch.diag_embed(torch.diagonal(sim))
                loss += torch.sum(off_diag ** 2) / (
                    class_features.size(0) * features.size(1)
                )
        return loss / len(unique_labels)