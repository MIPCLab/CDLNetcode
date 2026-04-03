import torch
import torch.nn.functional as F


def compute_entropy(logits):
    """计算每个样本的预测熵，返回 [B, 1]。"""
    probs = F.softmax(logits, dim=1)
    return -torch.sum(probs * torch.log(probs + 1e-9), dim=1, keepdim=True)


def calculate_FDI(image_features):
    """计算特征扩散指数（FDI），输入 [N, C, H, W]，输出 [H, W]。"""
    N, C, H, W = image_features.shape
    image_features = image_features.view(N, C, -1)
    mean_feature = image_features.mean(dim=0)
    squared_diff = (image_features - mean_feature) ** 2
    return squared_diff.sum(dim=1).view(H, W)


def calculate_local_entropy(image, block_h, block_w):
    """
    计算图像每个 block_h x block_w 块的局部熵。
    输入 image: [C, H, W]，输出 entropy_map: [num_blocks, num_blocks]。
    """
    _, H, W = image.shape
    num_blocks = H // block_h
    entropy_map = torch.zeros((num_blocks, num_blocks), dtype=torch.float32)
    for i in range(num_blocks):
        for j in range(num_blocks):
            block = image[:, i * block_h:(i + 1) * block_h,
                             j * block_w:(j + 1) * block_w]
            block = block.view(block.size(0), -1)
            _, counts = block.unique(return_counts=True)
            probs = counts.float() / block.numel()
            entropy_map[i, j] = -torch.sum(probs * torch.log(probs + 1e-6))
    return entropy_map