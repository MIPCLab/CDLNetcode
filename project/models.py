import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ──────────────────────────────────────────────────────────────
# 子模块
# ──────────────────────────────────────────────────────────────
def generate_dynamic_mask(image_size, image_features, num_blocks=4, mask_ratio=0.2, lambda1=1.0, lambda2=1.0):
    """
    动态生成4x4分块的掩膜，根据类别特征扩散指数（FDI）和局部熵（H_local）调整每个块的掩膜。
    :param image_size: 输入图片的形状 (C, H, W)
    :param image_features: 输入图像的特征图，形状为 (N, C, H, W)
    :param num_blocks: 将图片划分为 num_blocks x num_blocks 的网格
    :param mask_ratio: 掩膜的比例（不超过此比例）
    :param lambda1: 调节类别特征扩散指数（FDI）的权重
    :param lambda2: 调节局部熵（H_local）的权重
    :return: 掩膜张量，形状与输入图片一致
    """
    _, H, W = image_size
    block_h, block_w = H // num_blocks, W // num_blocks

    # 计算类别特征扩散指数 FDI
    FDI_map = calculate_FDI(image_features)

    # 计算局部熵
    entropy_map = calculate_local_entropy(image_size, block_h, block_w)

    # 根据 FDI 和局部熵生成掩膜概率图
    mask_prob_map = torch.sigmoid(lambda1 * FDI_map + lambda2 * entropy_map)

    # 计算总块数和最大掩盖块数
    total_blocks = num_blocks * num_blocks
    max_mask_blocks = int(mask_ratio * total_blocks)

    # 随机选择掩盖的块索引，基于掩膜概率进行选择
    masked_indices = []
    for i in range(num_blocks):
        for j in range(num_blocks):
            if random.random() > mask_prob_map[i, j]:  # 基于概率决定是否掩盖
                masked_indices.append(i * num_blocks + j)

    # 初始化掩膜网格
    mask_grid = torch.ones((num_blocks, num_blocks), dtype=torch.float32)

    for idx in masked_indices[:max_mask_blocks]:
        row = idx // num_blocks
        col = idx % num_blocks
        mask_grid[row, col] = 0  # 掩盖该块

    # 将掩膜网格扩展到输入图像的尺寸
    mask = torch.zeros((H, W), dtype=torch.float32)
    for i in range(num_blocks):
        for j in range(num_blocks):
            mask[i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w] = mask_grid[i, j]

    return mask.unsqueeze(0)  # 添加通道维度，与输入图像形状匹配
class GatedSelfAttention(nn.Module):
    """
    门控自注意力模块。
    对 use_entropy_gate=True 的样本启用门控增强；
    对其余样本直接透传输入特征 x。
    """

    def __init__(self, in_channels, out_channels, entropy_threshold=1.8):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.gate_conv  = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.gate_strength_fc = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1),
        )
        self.softmax = nn.Softmax(dim=-1)
        self.entropy_threshold = entropy_threshold

    def forward(self, x, use_entropy_gate):
        B, C, H, W = x.size()

        # 随机稀疏掩码，增加注意力鲁棒性
        mask     = torch.rand((B, 1, H, W), device=x.device) > 0.8
        masked_x = x * mask
        # masks = torch.stack([generate_block_mask((C, H, W), self.num_blocks, self.mask_ratio) for _ in range(batch_size)]).to(x.device)
        # masked_x = x * masks

        query = self.query_conv(masked_x).view(B, -1, H * W).permute(0, 2, 1)
        key   = self.key_conv(masked_x).view(B, -1, H * W)
        value = self.value_conv(x).view(B, -1, H * W)

        attention = self.softmax(torch.bmm(query, key))
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(B, C, H, W)

        gate_strength = torch.sigmoid(self.gate_strength_fc(x))
        gate          = torch.sigmoid(self.gate_conv(x))

        out[use_entropy_gate]  = (
            out[use_entropy_gate]
            * gate[use_entropy_gate]
            * gate_strength[use_entropy_gate]
        )
        out[~use_entropy_gate] = x[~use_entropy_gate]
        return out


class GroupFC(nn.Module):
    """
    分组全连接层：将特征等分为 num_groups 组，
    每组独立映射到 num_classes，最终结果相加。
    """

    def __init__(self, in_features, num_classes, num_groups=3):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = in_features // num_groups
        self.group_fc = nn.ModuleList([
            nn.Linear(self.group_size, num_classes)
            for _ in range(num_groups)
        ])

    def forward(self, x):
        return sum(
            self.group_fc[i](x[:, i * self.group_size:(i + 1) * self.group_size])
            for i in range(self.num_groups)
        )


class RegionSelector(nn.Module):
    """
    在 4×4 网格上用滑动窗口（window_grid_size×window_grid_size）
    选出响应最高的 top_k 个区域，返回左上角网格坐标。
    """

    def __init__(self, grid_size=4, window_grid_size=3, top_k=1):
        super().__init__()
        self.grid_size        = grid_size
        self.window_grid_size = window_grid_size
        self.top_k            = top_k

    def forward(self, sampling_map):
        B, C, H, W = sampling_map.size()
        grid_h = H // self.grid_size
        grid_w = W // self.grid_size

        # 将采样图划分为 grid_size×grid_size 个格子，取各格均值
        grid_responses = (
            sampling_map
            .unfold(2, grid_h, grid_h)
            .unfold(3, grid_w, grid_w)
            .contiguous()
            .view(B, C, self.grid_size, self.grid_size, -1)
            .mean(dim=-1)
        )

        # 滑动窗口求和，选 top_k
        unfolded    = F.unfold(grid_responses, kernel_size=self.window_grid_size,
                               stride=1, padding=0)
        window_sums = unfolded.sum(dim=1)
        _, topk_idx = window_sums.topk(self.top_k, dim=1)

        stride = self.grid_size - self.window_grid_size + 1
        topk_coords = torch.zeros(B, self.top_k, 2, dtype=torch.long,
                                  device=sampling_map.device)
        for b in range(B):
            for k in range(self.top_k):
                idx = topk_idx[b, k]
                topk_coords[b, k, 0] = idx // stride
                topk_coords[b, k, 1] = idx % stride
        return topk_coords  # [B, top_k, 2]


class FeatureExtractor(nn.Module):
    """
    根据 RegionSelector 选出的窗口位置裁剪原图，
    送入 backbone 提取细化特征，并生成 refined_response_maps。
    """

    def __init__(self, backbone1, backbone2,
                 window_grid_size=3, grid_size=4, resize_size=(256, 256)):
        super().__init__()
        self.backbone1        = backbone1
        self.backbone2        = backbone2
        self.window_grid_size = window_grid_size
        self.grid_size        = grid_size
        self.resize_size      = resize_size

    def forward(self, images, window_positions):
        B, C, H, W = images.size()
        grid_h = H // self.grid_size
        grid_w = W // self.grid_size
        top_k  = window_positions.size(1)

        cropped_images = []
        for b in range(B):
            for k in range(top_k):
                row_start, col_start = window_positions[b, k]
                y_start = max(int(row_start) * grid_h, 0)
                y_end   = min(y_start + self.window_grid_size * grid_h, H)
                x_start = max(int(col_start) * grid_w, 0)
                x_end   = min(x_start + self.window_grid_size * grid_w, W)
                crop = images[b:b + 1, :, y_start:y_end, x_start:x_end]
                crop_resized = F.interpolate(crop, size=self.resize_size,
                                             mode='bilinear', align_corners=False)
                cropped_images.append(crop_resized)

        if cropped_images:
            cropped_images  = torch.cat(cropped_images, dim=0)
            features        = self.backbone2(self.backbone1(cropped_images))
            refined_features = F.interpolate(features, size=(H, W),
                                             mode='bilinear', align_corners=False)
        else:
            refined_features = torch.zeros(B * top_k, 2048, H, W, device=images.device)
            features         = refined_features

        response_head          = nn.Conv2d(2048, 1, kernel_size=1).to(images.device)
        refined_response_maps  = (torch.sigmoid(response_head(refined_features))
                                  .view(B, top_k, 1, H, W))
        crop_imgs = features * torch.sigmoid(response_head(features))
        return refined_response_maps, crop_imgs  # [B, top_k, 1, H, W]


class FeatureFuser(nn.Module):
    """
    将 FeatureExtractor 输出的 refined_response_maps
    融合（替换）回原始 sampling_map 对应区域。
    """

    def __init__(self, grid_size=4, window_grid_size=3):
        super().__init__()
        self.grid_size        = grid_size
        self.window_grid_size = window_grid_size

    def forward(self, sampling_map, refined_response_maps, selected_regions):
        B, C, H, W = sampling_map.size()
        top_k  = refined_response_maps.size(1)
        grid_h = H // self.grid_size
        grid_w = W // self.grid_size

        fused = sampling_map.clone()
        for b in range(B):
            for k in range(top_k):
                row_start, col_start = selected_regions[b, k]
                y_start = max(int(row_start) * grid_h, 0)
                y_end   = min(y_start + self.window_grid_size * grid_h, H)
                x_start = max(int(col_start) * grid_w, 0)
                x_end   = min(x_start + self.window_grid_size * grid_w, W)
                fused[b, :, y_start:y_end, x_start:x_end] = (
                    refined_response_maps[b, k, :, y_start:y_end, x_start:x_end]
                )
        return torch.sigmoid(fused)  # [B, 1, H, W]


# ──────────────────────────────────────────────────────────────
# 主模型
# ──────────────────────────────────────────────────────────────

class PSAResNet(nn.Module):
    """
    PSA-ResNet50 细粒度分类主干。

    前向输出（5 项）：
        output              分类 logits        [B, num_classes]
        sampling_map        采样权重图          [B, 1, H', W']
        pooled_feature      fused 全局特征      [B, 2048]
        fused_sampling_map  融合后下采样权重图  [B, 1, H', W']
        binary_logit        难/易类别二分类     [B, 1]

    binary_head 以 raw_pooled || fused_pooled（共 4096-d）为输入，
    无需额外的大型二分类网络，同时保留了判别性区域信息。
    """

    def __init__(self, num_classes=23):
        super().__init__()
        resnet = models.resnet50(pretrained=True)

        self.backbone1   = nn.Sequential(*list(resnet.children())[:6])
        self.backbone2   = nn.Sequential(*list(resnet.children())[6:8])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # proxy 相关参数
        self.num_proxies = 3
        self.num_classes = num_classes
        self.proxies     = nn.Parameter(torch.randn(num_classes * self.num_proxies, 2048))
        self.alpha = self.beta = self.gamma = 1

        # 核心子模块
        self.sampling_head    = nn.Conv2d(2048, 1, kernel_size=1)
        self.region_selector  = RegionSelector(grid_size=4, window_grid_size=3, top_k=1)
        self.feature_extractor = FeatureExtractor(
            self.backbone1, self.backbone2,
            window_grid_size=3, grid_size=4, resize_size=[256, 256],
        )
        self.feature_fuser  = FeatureFuser(grid_size=4, window_grid_size=3)
        self.channel_gate   = GatedSelfAttention(2048, 2048, entropy_threshold=1.5)
        self.group_fc       = GroupFC(in_features=2048, num_classes=num_classes, num_groups=3)

        # 轻量二分类头：输入 raw_pooled(2048) + fused_pooled(2048) = 4096
        self.binary_head = nn.Sequential(
            nn.Linear(4096, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    # ── forward ───────────────────────────────────────────────

    def forward(self, x, use_entropy_gate=torch.tensor([])):
        B, C, H, W = x.size()

        # 1. 骨干特征提取
        f2 = self.backbone1(x)
        f4 = self.backbone2(f2)
        f4 = self.channel_gate(f4, use_entropy_gate)

        # 2. 采样权重图（含反事实对比）
        sampling_map       = torch.sigmoid(self.sampling_head(f4))
        counterfactual_map = 1 - sampling_map
        sampling_map       = torch.sigmoid(sampling_map - counterfactual_map)

        # 3. 区域选择 → 细化 → 融合
        upsampled_map      = F.interpolate(sampling_map, size=(H, W),
                                           mode='bilinear', align_corners=False)
        selected_regions   = self.region_selector(upsampled_map)
        refined_maps, _    = self.feature_extractor(x, selected_regions)
        fused_map          = self.feature_fuser(upsampled_map, refined_maps, selected_regions)

        # 4. 特征加权 + 全局池化
        fused_map_down  = F.interpolate(fused_map, size=(f4.size(2), f4.size(3)),
                                        mode='bilinear', align_corners=False)
        fused_features  = f4 * fused_map_down
        pooled_feature  = self.global_pool(fused_features).view(B, -1)   # [B, 2048]

        # 5. 分类输出
        output = self.group_fc(pooled_feature)

        # 6. 二分类输出（raw + fused 拼接，特征互补）
        raw_pooled   = self.global_pool(f4).view(B, -1)                   # [B, 2048]
        binary_feat  = torch.cat([raw_pooled, pooled_feature], dim=1)     # [B, 4096]
        binary_logit = self.binary_head(binary_feat)                      # [B, 1]

        return output, sampling_map, pooled_feature, fused_map_down, binary_logit

    # ── 损失函数 ───────────────────────────────────────────────

    def compute_push_loss(self, features1, features2, labels):
        """Push-out loss：同图不同增强拉近，不同图推远。"""
        batch_size = features1.size(0)
        agg_loss   = torch.mean(1 - F.cosine_similarity(features1, features2))
        sep_loss   = sum(
            -torch.mean(F.cosine_similarity(features1[i].unsqueeze(0),
                                            features2[j].unsqueeze(0)))
            for i in range(batch_size)
            for j in range(batch_size)
            if i != j
        )
        sep_loss /= batch_size * (batch_size - 1)
        return agg_loss, sep_loss

    def compute_pull_loss(self, features, labels):
        """Pull-in loss：向 proxy 聚拢，远离他类 proxy。"""
        batch_size = features.size(0)
        agg_loss = sep_loss = proxy_loss = 0.0

        for i in range(batch_size):
            label = labels[i]
            f     = features[i]
            class_proxies = self.proxies[
                label * self.num_proxies:(label + 1) * self.num_proxies
            ]
            agg_loss += torch.mean(1 - F.cosine_similarity(f.unsqueeze(0), class_proxies))
            other_proxies = torch.cat([
                self.proxies[:label * self.num_proxies],
                self.proxies[(label + 1) * self.num_proxies:],
            ])
            sep_loss += torch.mean(
                F.relu(F.cosine_similarity(f.unsqueeze(0), other_proxies))
            )

        proxies = self.proxies.view(self.num_classes, self.num_proxies, -1)
        for proxy_set in proxies:
            for i in range(len(proxy_set)):
                for j in range(i + 1, len(proxy_set)):
                    proxy_loss += F.relu(
                        F.cosine_similarity(
                            proxy_set[i].unsqueeze(0),
                            proxy_set[j].unsqueeze(0),
                        ) - 0.5
                    )

        return (agg_loss / batch_size,
                sep_loss / batch_size,
                proxy_loss / len(self.proxies))

    def compute_total_loss(self, features1, features2, logits, labels):
        agg_push, sep_push         = self.compute_push_loss(features1, features2, labels)
        agg_pull, sep_pull, proxy  = self.compute_pull_loss(features1, labels)
        return (
            self.alpha * (agg_push + sep_push)
            + self.beta  * (agg_pull + sep_pull + proxy)
        )