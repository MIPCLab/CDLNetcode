import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from collections import Counter,defaultdict
import cv2
from torch.optim import SGD
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
# from sklearn.decomposition import IncrementalPCA
# from causal import Causal_Norm_Classifier
# from sklearn.decomposition import PCA
import seaborn as sns
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
import random
from torchvision.ops import deform_conv2d
from dataloader import get_dataloader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report
from torchvision.transforms import ToPILImage
import copy
from PIL import Image
import torch.nn.functional as F
import os
os.environ['CUDA_VISIBLE_DEVICES']='2,3,1'

def calculate_FDI(image_features):
    """
    计算类别特征扩散指数（FDI），通过计算每个像素块的特征与全图均值特征的差异。
    :param image_features: 形状为 (N, C, H, W) 的特征图，N 是样本数，C 是通道数，H 和 W 是图像高度和宽度
    :return: 类别特征扩散指数 FDI，形状为 (H, W)
    """
    N, C, H, W = image_features.shape
    # 将图像特征展平为每个像素块的特征向量
    image_features = image_features.view(N, C, -1)  # 形状变为 (N, C, H*W)
    
    # 计算图像特征的均值
    mean_feature = image_features.mean(dim=0)  # 形状为 (C, H*W)

    # 计算每个像素块的特征差异 (均方误差)
    squared_diff = (image_features - mean_feature) ** 2
    FDI_map = squared_diff.sum(dim=1).view(H, W)  # 计算每个像素块的差异

    return FDI_map
def calculate_local_entropy(image, block_h, block_w):
    """
    计算图像每个4x4块的局部熵。
    :param image: 输入图像，形状为 (C, H, W)
    :param block_h: 每个块的高度
    :param block_w: 每个块的宽度
    :return: 每个块的熵值，形状为 (num_blocks, num_blocks)
    """
    _, H, W = image.shape
    num_blocks = H // block_h

    entropy_map = torch.zeros((num_blocks, num_blocks), dtype=torch.float32)

    for i in range(num_blocks):
        for j in range(num_blocks):
            # 获取每个块的像素
            block = image[:, i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w]
            block = block.view(block.size(0), -1)  # 拉平成二维
            _, counts = block.unique(return_counts=True)  # 统计每个像素值的出现次数
            probs = counts.float() / block.numel()  # 计算概率分布
            entropy = -torch.sum(probs * torch.log(probs + 1e-6))  # 计算熵
            entropy_map[i, j] = entropy

    return entropy_map

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
    def __init__(self, in_channels, out_channels, entropy_threshold=1.8):
        super(GatedSelfAttention, self).__init__()

        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.gate_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # 动态门控强度生成层
        self.gate_strength_fc = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1)  # 输出自适应门控强度
        )

        self.num_blocks = 4
        self.mask_ratio = 0.2
        self.softmax = nn.Softmax(dim=-1)
        self.entropy_threshold = entropy_threshold  # 控制是否启用门控的熵阈值

    def forward(self, x, use_entropy_gate):
        batch_size, C, H, W = x.size()
        mask = torch.rand((batch_size, 1, H, W), device=x.device)>0.8
        masked_x = x * mask  # 应用掩码
        # masks = torch.stack([generate_block_mask((C, H, W), self.num_blocks, self.mask_ratio) for _ in range(batch_size)]).to(x.device)
        # masked_x = x * masks

        # 自注意力计算
        query = self.query_conv(masked_x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.key_conv(masked_x).view(batch_size, -1, H * W)
        value = self.value_conv(x).view(batch_size, -1, H * W)

        attention = self.softmax(torch.bmm(query, key))  # 计算注意力权重
        out = torch.bmm(value, attention.permute(0, 2, 1))  # 加权输出
        out = out.view(batch_size, C, H, W)


        # 判断是否启用门控增强：当熵高于阈值时启用门控
        gate_strength = torch.sigmoid(self.gate_strength_fc(x))
        gate = torch.sigmoid(self.gate_conv(x))

        # 应用门控条件
        
        out[use_entropy_gate] = out[use_entropy_gate] * gate[use_entropy_gate] * gate_strength[use_entropy_gate]  # 根据门控条件增强或保持原样
        out[~use_entropy_gate]=x[~use_entropy_gate]


        return out
    
class GroupFC(nn.Module):
    def __init__(self, in_features, num_classes, num_groups=3):
        super(GroupFC, self).__init__()
        self.num_groups = num_groups
        self.group_size = in_features // num_groups
        self.group_fc = nn.ModuleList([
            nn.Linear(self.group_size, num_classes) for _ in range(num_groups)
        ])

    def forward(self, x):
        # 将输入特征分组
        group_outputs = []
        for i in range(self.num_groups):
            group_input = x[:, i * self.group_size:(i + 1) * self.group_size]
            group_output = self.group_fc[i](group_input)
            group_outputs.append(group_output)
        # 将各组的输出相加
        final_output = sum(group_outputs)
        return final_output

class RegionSelector(nn.Module):
    def __init__(self, grid_size=4, window_grid_size=3, top_k=1):
        super(RegionSelector, self).__init__()
        self.grid_size = grid_size
        self.window_grid_size = window_grid_size
        self.top_k = top_k

    def forward(self, sampling_map):
        """
        选择响应值最高的3x3网格区域。

        Args:
            sampling_map (Tensor): 上采样后的采样权重图，形状 [B, 1, H, W]

        Returns:
            selected_windows (Tensor): 选定的窗口位置，形状 [B, top_k, 2]，每个位置包含 (row, col)
        """
        B, C, H, W = sampling_map.size()
        grid_size = self.grid_size
        window_grid_size = self.window_grid_size
        top_k = self.top_k

        grid_h = H // grid_size
        grid_w = W // grid_size

        # 将采样图划分为4x4网格，并计算每个网格的平均响应
        grid_responses = sampling_map.unfold(2, grid_h, grid_h).unfold(3, grid_w, grid_w)  # [B, 1, 4, 4, grid_h, grid_w]
        grid_responses = grid_responses.contiguous().view(B, C, grid_size, grid_size, -1).mean(dim=-1)  # [B, 1, 4, 4]

        # 使用滑动窗口选择3x3的网格区域，计算每个3x3区域的总响应
        window_size = window_grid_size
        stride = 1
        padding = 0
        unfolded = F.unfold(grid_responses, kernel_size=window_size, stride=stride, padding=padding)  # [B, window_size*window_size, L]
        # Sum the responses within each 3x3 window
        window_sums = unfolded.sum(dim=1)  # [B, L]
        # Number of possible 3x3 windows in 4x4 grid: 2x2=4

        # Select top_k windows with highest sums
        topk_vals, topk_idx = window_sums.topk(top_k, dim=1)  # [B, top_k]

        # Convert window indices to grid coordinates
        topk_coords = torch.zeros(B, top_k, 2, dtype=torch.long, device=sampling_map.device)  # [B, top_k, 2]
        for b in range(B):
            for k in range(top_k):
                idx = topk_idx[b, k]
                row = idx // (grid_size - window_grid_size + 1)
                col = idx % (grid_size - window_grid_size + 1)
                topk_coords[b, k, 0] = row
                topk_coords[b, k, 1] = col

        return topk_coords  # [B, top_k, 2]
class FeatureExtractor(nn.Module):
    def __init__(self, backbone1, backbone2, window_grid_size=3, grid_size=4, resize_size=(224, 224)):
        super(FeatureExtractor, self).__init__()
        self.backbone1 = backbone1
        self.backbone2 = backbone2
        self.window_grid_size = window_grid_size
        self.grid_size = grid_size
        self.resize_size = resize_size

    def forward(self, images, window_positions):
        """
        根据选定的窗口位置裁剪图像，并提取细化后的特征。

        Args:
            images (Tensor): 原始图像，形状 [B, C, H, W]
            window_positions (Tensor): 选定的窗口位置，形状 [B, top_k, 2]

        Returns:
            refined_features (Tensor): 融合后的特征图，形状 [B, top_k, C, H, W]
        """
        B, C, H, W = images.size()
        grid_size = self.grid_size
        window_grid_size = self.window_grid_size
        grid_h = H // grid_size
        grid_w = W // grid_size
        top_k = window_positions.size(1)

        # 计算每个3x3网格窗口对应的像素坐标
        cropped_images = []
        for b in range(B):
            for k in range(top_k):
                row_start, col_start = window_positions[b, k]
                y_start = row_start * grid_h
                y_end = y_start + window_grid_size * grid_h
                x_start = col_start * grid_w
                x_end = x_start + window_grid_size * grid_w

                # 确保不超出边界
                y_start = max(y_start, 0)
                y_end = min(y_end, H)
                x_start = max(x_start, 0)
                x_end = min(x_end, W)

                crop = images[b:b+1, :, y_start:y_end, x_start:x_end]  # [1, C, window_grid_size*grid_h, window_grid_size*grid_w]
                # 调整大小
                crop_resized = F.interpolate(crop, size=self.resize_size, mode='bilinear', align_corners=False)  # [1, C, resize_H, resize_W]
                cropped_images.append(crop_resized)

        if cropped_images:
            cropped_images = torch.cat(cropped_images, dim=0)  # [B * top_k, C, resize_H, resize_W]
            # 提取特征
            features = self.backbone1(cropped_images)  # [B * top_k, 2048, H', W']
            features = self.backbone2(features)  # [B * top_k, 2048, H', W']

            # 调整特征图大小
            refined_features = F.interpolate(features, size=(H, W), mode='bilinear', align_corners=False)  # [B * top_k, 2048, H, W]
        else:
            # 如果没有裁剪区域，返回全零的特征图
            refined_features = torch.zeros(B * top_k, 2048, H, W).to(images.device)
        response_head = nn.Conv2d(2048, 1, kernel_size=1).to(images.device)
        refined_response_maps = torch.sigmoid(response_head(refined_features)) # [B * top_k, 1, H, W]
        refined_response_maps_crop = torch.sigmoid(response_head(features))
        crop_imgs = features*refined_response_maps_crop
        # 重新组织特征响应图
        refined_response_maps = refined_response_maps.view(B, top_k, 1, H, W)  # [B, top_k, 1, H, W]

        return refined_response_maps,crop_imgs  # [B, top_k, 1, H, W]
class FeatureFuser(nn.Module):
    def __init__(self, grid_size=4, window_grid_size=3):
        super(FeatureFuser, self).__init__()
        self.grid_size = grid_size
        self.window_grid_size = window_grid_size

    def forward(self, sampling_map, refined_response_maps, selected_regions):
        """
        将细化后的特征响应图融合回采样权重图。

        Args:
            sampling_map (Tensor): 上采样后的采样权重图，形状 [B, 1, H, W]
            refined_response_maps (Tensor): 细化后的特征响应图，形状 [B, top_k, 1, H, W]
            selected_regions (Tensor): 选定的窗口索引，形状 [B, top_k, 2]

        Returns:
            fused_sampling_map (Tensor): 融合后的采样权重图，形状 [B, 1, H, W]
        """
        B, C, H, W = sampling_map.size()
        top_k = refined_response_maps.size(1)
        grid_size = self.grid_size
        window_grid_size = self.window_grid_size

        grid_h = H // grid_size
        grid_w = W // grid_size

        fused_sampling_map = sampling_map.clone()  # [B,1,H,W]

        for b in range(B):
            for k in range(top_k):
                row_start, col_start = selected_regions[b, k]
                y_start = row_start * grid_h
                y_end = y_start + window_grid_size * grid_h
                x_start = col_start * grid_w
                x_end = x_start + window_grid_size * grid_w

                # 确保不超出边界
                y_start = max(y_start, 0)
                y_end = min(y_end, H)
                x_start = max(x_start, 0)
                x_end = min(x_end, W)

                # 获取对应区域的特征响应图
                window_response = refined_response_maps[b, k, :, y_start:y_end, x_start:x_end]  # [1, H', W']

                # 融合特征响应图到采样权重图中
                fused_sampling_map[b, :, y_start:y_end, x_start:x_end] = window_response   ###融合变为替换

        # 确保融合后的采样图在0-1之间
        fused_sampling_map = torch.sigmoid(fused_sampling_map)

        return fused_sampling_map  # [B,1,H,W]
class PSAResNet(nn.Module):
    def __init__(self, num_classes=23):
        super(PSAResNet, self).__init__()
        resnet = models.resnet50(pretrained=True)

        # 使用 ResNet 的 layer4
        self.backbone1 = nn.Sequential(*list(resnet.children())[:6])  # 使用 ResNet 前几层直到 layer4
        self.backbone2 = nn.Sequential(*list(resnet.children())[6:8])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_proxies = 3
        self.num_classes = 42
        self.proxies = nn.Parameter(torch.randn(num_classes * self.num_proxies, 2048))
        self.alpha = 1 # 推开损失权重
        self.beta = 1    # 拉近损失权重
        self.gamma = 1  # 分类损失权重
        # 定义生成采样权重图的卷积层，基于 layer4 的输出
        self.sampling_head = nn.Conv2d(2048, 1, kernel_size=1)  # layer4 的输出通道是 2048
        # 最终的分类层，基于 layer4 的输出
        self.region_selector = RegionSelector(grid_size=4, window_grid_size=3, top_k=1)
        self.feature_extractor = FeatureExtractor(self.backbone1, self.backbone2, window_grid_size=3, grid_size=4, resize_size=[256,256])
        self.feature_fuser = FeatureFuser(grid_size=4, window_grid_size=3)
        self.channel_gate = GatedSelfAttention(2048,2048,entropy_threshold=1.5)
        self.fc = nn.Linear(2048, num_classes)
        self.grid_size = 4
        self.top_k = 1
        self.resize_size = (256,256)
        self.transform = transforms.Compose([
            transforms.Resize(self.resize_size),
            transforms.ToTensor(),
        ])
        self.group_fc = GroupFC(in_features=2048, num_classes=num_classes, num_groups=3)


    def forward(self, x,use_entropy_gate=torch.tensor([])):
        B, C, H, W = x.size()

        # Step 1: 提取特征
        f2 = self.backbone1(x)  # [B, 2048, H', W']
        f4 = self.backbone2(f2)  # [B, 2048, H', W']

        # Step 2: 生成采样权重图
        f4 = self.channel_gate(f4,use_entropy_gate)
        sampling_map = torch.sigmoid(self.sampling_head(f4))  # [B, 1, H', W']
        counterfactual_map = 1 - sampling_map
        image_difference = sampling_map - counterfactual_map
        sampling_map = torch.sigmoid(image_difference)  # [B, 1, H', W']
        # Step 3: 上采样 sampling_map 到原图大小
        upsampled_sampling_map = F.interpolate(sampling_map, size=(H, W), mode='bilinear', align_corners=False)  # [B,1,H,W]

        # Step 4: 选择区域
        selected_regions = self.region_selector(upsampled_sampling_map)  # [B, top_k, 2]

        # Step 5: 裁剪并提取细化后的特征响应图
        refined_response_maps,crop_imgs = self.feature_extractor(x, selected_regions)  # [B, top_k, 1, H, W]

        # Step 6: 特征融合
        fused_sampling_map = self.feature_fuser(upsampled_sampling_map, refined_response_maps, selected_regions)  # [B,1,H,W]
        # Step 7: 将 fused_sampling_map 与原始特征图 f4 相乘
        # 需要将 fused_sampling_map 下采样到 f4 的尺寸
        fused_sampling_map_down = F.interpolate(fused_sampling_map, size=(f4.size(2), f4.size(3)), mode='bilinear', align_corners=False)  # [B,1,H',W']
        # sampling_map_bias=None
        # if sum(use_entropy_gate)>0:
        #     sampling_map_bias =generate_attention_maps(sampling_map[use_entropy_gate],device)
        #     weighted_features = []  # 存储加权特征图
        #     for attention in sampling_map_bias:
        #         attention = attention.unsqueeze(1)  # (B, 1, H, W)
        #         weighted_feature = f4[use_entropy_gate] * attention  # 逐像素相乘
        #         weighted_features.append(weighted_feature)

        #     # 对每个加权特征图进行全局平均池化，并拼接成一个长向量
        #     pooled_features = [F.adaptive_avg_pool2d(wf, (1, 1)).view(f4[use_entropy_gate].size(0), -1) for wf in weighted_features]

        #     # 重新组织 pooled_features 成 (B * C, num_attention_maps)
        #     B, C = f4[use_entropy_gate].size(0), pooled_features[0].size(1)
        #     num_attention_maps = len(pooled_features)
        #     # combined_features = torch.stack(pooled_features, dim=2)  # (B, C, num_attention_maps)
        #     combined_features = torch.cat((pooled_features[0],pooled_features[1],pooled_features[2]),dim=0)
        #     # combined_features = combined_features.view(B * num_attention_maps,C) 
        
        fused_features = f4 * fused_sampling_map_down   # [B,2048,H',W']

        # Step 8: 全局池化和分类
        pooled_feature = self.global_pool(fused_features)  # [B, 2048, 1, 1]
        pooled_feature = pooled_feature.view(pooled_feature.size(0), -1)  # [B, 2048]
        # if sum(use_entropy_gate)>0:
        #     # 将 pooled_feature 和 combined_features 拼接在一起
        #     final_features = torch.cat([pooled_feature, combined_features], dim=0)  # (B, 2048 + C * num_attention_maps)
        #     output = self.group_fc(final_features)

        #     return output, sampling_map_bias, pooled_feature 
        # else:
        final_features = pooled_feature
        output = self.group_fc(final_features)

        return output, sampling_map, pooled_feature ,fused_sampling_map_down

        # Step 9: 分类
        # output = self.fc(pooled_feature)  # [B, num_classes]
         # 返回分类结果和采样权重图
   

    
    def crop_selected_grids(self, images, window_positions, grid_size=4, window_grid_size=3, resize_size=(224, 224)):
        """
        从原图中裁剪选定的3x3网格区域并调整大小
        images: [B, C, H, W]
        window_positions: [B, topk, 2] 每个位置包含 (row, col) 的网格索引
        grid_size: 总网格数量（4）
        window_grid_size: 选择的网格窗口大小（3）
        resize_size: 调整大小后的尺寸
        返回裁剪并调整大小后的图像：[B * topk * 9, C, resize_H, resize_W]
        """
        B, C, H, W = images.size()
        grid_h = H // grid_size
        grid_w = W // grid_size
        cropped_images = []
        for b in range(B):
            for k in range(window_positions.size(1)):  # topk
                row, col = window_positions[b, k]
                for m in range(window_grid_size):
                    for n in range(window_grid_size):
                        current_row = row + m
                        current_col = col + n
                        # 确保不超出边界
                        current_row = min(current_row, grid_size - 1)
                        current_col = min(current_col, grid_size - 1)
                        y_start = current_row * grid_h
                        y_end = y_start + grid_h
                        x_start = current_col * grid_w
                        x_end = x_start + grid_w
                        crop = images[b, :, y_start:y_end, x_start:x_end]  # [C, grid_h, grid_w]
                        crop = F.interpolate(crop.unsqueeze(0), size=resize_size, mode='bilinear', align_corners=False)  # [1, C, resize_H, resize_W]
                        cropped_images.append(crop.squeeze(0))  # [C, resize_H, resize_W]
        if cropped_images:
            cropped_images = torch.stack(cropped_images)  # [B * topk * 9, C, resize_H, resize_W]
        else:
            # 如果没有裁剪区域，返回空的 tensor
            cropped_images = torch.empty(0, C, resize_size[0], resize_size[1]).to(images.device)
        return cropped_images
    def compute_push_loss(self, features1, features2,labels):
        """
        Push-out loss: Separate different images and aggregate augmented views of the same image.
        """
        batch_size, feat_dim = features1.size()

        # Push-out losses
        agg_loss, sep_loss = 0, 0

        for i in range(batch_size):
            label = labels[i]
            # Aggregation loss (same image, different augmentations)
            agg_loss += torch.mean(1 - F.cosine_similarity(features1[i].unsqueeze(0), features2[i].unsqueeze(0)))

            # Separation loss (different images)
            for j in range(batch_size):
                if i != j:
                    sep_loss += -torch.mean(F.cosine_similarity(features1[i].unsqueeze(0), features2[j].unsqueeze(0)))

        # Normalize losses
        agg_loss /= batch_size
        sep_loss /= (batch_size * (batch_size - 1))  # Avoid double counting

        return agg_loss, sep_loss

    def compute_pull_loss(self, features, labels):
        """
        Pull-in loss: Aggregate towards proxies and separate from non-proxies.
        """
        batch_size, feat_dim = features.size()
        agg_loss, sep_loss, proxy_loss = 0, 0, 0

        for i in range(batch_size):
            label = labels[i]
            feature = features[i]
            # Get proxies of the same class
            class_proxies = self.proxies[label * self.num_proxies:(label + 1) * self.num_proxies]

            # Aggregation loss
            agg_loss += torch.mean(1 - F.cosine_similarity(feature.unsqueeze(0), class_proxies))

            # Separation loss
            other_proxies = torch.cat([
                self.proxies[:label * self.num_proxies],
                self.proxies[(label + 1) * self.num_proxies:]
            ])
            sep_loss += torch.mean(F.relu(F.cosine_similarity(feature.unsqueeze(0), other_proxies)))

        # Proxy regularization: Prevent overlap of proxies
        proxies = self.proxies.view(self.num_classes, self.num_proxies, -1)  # Reshape proxies into [num_classes, num_proxies, feature_dim]
        for proxy_set in proxies:  # Iterate over each class's proxies
            for i in range(len(proxy_set)):
                for j in range(i + 1, len(proxy_set)):
                    proxy_loss += F.relu(F.cosine_similarity(proxy_set[i].unsqueeze(0), proxy_set[j].unsqueeze(0)) - 0.5)

        # Normalize losses
        agg_loss /= batch_size
        sep_loss /= batch_size
        proxy_loss /= len(self.proxies)

        return agg_loss, sep_loss, proxy_loss

    def compute_total_loss(self, features1, features2, logits, labels):
        """
        Compute the total loss combining Push, Pull, and Classification losses.
        """
        # Push-out loss
        agg_loss_push, sep_loss_push = self.compute_push_loss(features1, features2,labels)

        # Pull-in loss
        agg_loss_pull, sep_loss_pull, proxy_loss_pull = self.compute_pull_loss(features1, labels)

        # Classification loss
        # classification_loss = F.cross_entropy(logits, labels)

        # Total loss
        total_loss = (
            self.alpha * (agg_loss_push + sep_loss_push) +
            self.beta * (agg_loss_pull + sep_loss_pull + proxy_loss_pull) 
        )
        return total_loss
def compute_entropy(logits):
    """
    计算每个样本的熵值。
    参数:
    - logits: 模型输出的 logits (batch_size, num_classes)
    返回:
    - entropy: 每个样本的熵值 (batch_size,)
    """
    probs = F.softmax(logits, dim=1)  # 转换为概率
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1, keepdim=True)  # 熵公式
    return entropy

class FeatureDecouplingLoss(nn.Module):
    def __init__(self):
        super(FeatureDecouplingLoss, self).__init__()
        
    def forward(self, features, labels):
        loss = 0.0
        # class_weights={5: 0.05, 13: 0.05, 19: 0.05, 20:0.05}
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            # 获取当前类别的样本特征
            mask = labels == label
            class_features = features[mask]  # 形状为 (N, feature_dim)
            
            if class_features.size(0) > 1:  # 至少需要两个样本
                # 对特征向量进行归一化
                class_features = F.normalize(class_features, p=2, dim=-1)
                
                # 计算特征间的内积
                similarity_matrix = torch.matmul(class_features, class_features.t())  # (N, N)
                
                # 排除对角线元素 (自身的内积)
                off_diag = similarity_matrix - torch.diag_embed(torch.diagonal(similarity_matrix))
                
                # 对每个类别的损失进行归一化
                loss += torch.sum(off_diag ** 2) / (class_features.size(0) * features.size(1))  # 特征空间维度归一化
        
                # class_weight = class_weights.get(int(label.item()), 1.0)  # 如果未定义权重，默认1.0
                # loss += class_weight*class_loss
        return loss / len(unique_labels)  # 平均到类别数量

# 假设输入为特征图 `features` (B, C, H, W)
from timm.models import convnext_large
class BinaryConvNeXt(nn.Module):
    def __init__(self):
        super(BinaryConvNeXt, self).__init__()
        self.convnext = convnext_large(pretrained=True)
        num_ftrs = self.convnext.head.fc.in_features
        self.convnext.head.fc = nn.Linear(num_ftrs, 1)  # 二分类任务，输出1个值

    def forward(self, x):
        return self.convnext(x)

from efficientnet_pytorch import EfficientNet

# 定义二分类EfficientNet模型
class BinaryEfficientNet(nn.Module):
    def __init__(self):
        super(BinaryEfficientNet, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        num_ftrs = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Linear(num_ftrs, 1)  # 二分类任务，输出1个值
        

    def forward(self, x):
        logits = self.efficientnet(x)
        return logits

class BinaryEfficientNet6(nn.Module):
    def __init__(self):
        super(BinaryEfficientNet6, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b6')
        num_ftrs = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Linear(num_ftrs, 1)  # 二分类任务，输出1个值
        

    def forward(self, x):
        logits = self.efficientnet(x)
        return logits

# 定义二分类损失函数
binary_criterion = nn.BCEWithLogitsLoss()

def compute_fdi(features,mu_c, labels, c, alpha=0.9, use_moving_average=True):
    """
    计算类别 c 的特征扩散指数（FDI），可以选择是否使用滑动均值更新
    
    参数:
    features: 一个形状为 (N, D) 的 numpy 数组，其中 N 是样本数量，D 是特征的维度。
    labels: 一个形状为 (N,) 的 numpy 数组，表示每个样本所属的类别。
    c: 要计算 FDI 的类别标签。
    alpha: 动量系数，默认值为 0.9，仅在 use_moving_average=True 时有效。
    use_moving_average: 是否使用滑动均值更新，默认 False。
    
    返回:
    FDI(c): 特征扩散指数
    """
    # 提取类别 c 的样本
    features_c = features[labels == c]
    
    # 样本数量
    N_c = features_c.shape[0]
    
    # 初始化滑动均值 μ_c
    mu_c = mu_c[c]
    
    # 计算 FDI 并更新 μ_c
    fdi_c = 0
    
    for i in range(N_c):
        # 当前样本的特征
        x_i_c = features_c[i]
        
        if use_moving_average:
            # 计算类别 c 的滑动均值 μ_c(t) 更新
            mu_c = alpha * mu_c + (1 - alpha) * x_i_c
        else:
            # 如果不使用滑动均值，直接计算均值
            mu_c = np.mean(features_c, axis=0)
        
        # 计算当前样本的 FDI 部分
        fdi_c += np.linalg.norm(x_i_c - mu_c) ** 2
    
    # 计算最终的 FDI(c)
    fdi_c /= N_c  # 除以样本数，计算最终的平均值
    return fdi_c

def warmup_train(model, train_loader, optimizer_warmup,scheduler_warmup, classification_criterion, device, num_epochs=20,warmup=5):
    model.train()
    decoupling_loss_fn = FeatureDecouplingLoss()
    class_nums = 23
    best_accuracy = 0.0  # 记录最佳准确率
    best_model_path = "best_model.pth"  # 保存最佳模型的路径
    classes_to_enhance = set()
    class_frequency = defaultdict(int)
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        classes_to_enhance=[]
        class_entropy_sum_s = defaultdict(float)  # 记录每个类别的熵值总和
        class_sample_count_s = defaultdict(int)
        class_entropy_sum_w = defaultdict(float)  # 记录每个类别的熵值总和
        class_sample_count_w = defaultdict(int)
        # mu_c_w = {label: 0 for label in class_nums}
        # mu_c_s =  {label: 0 for label in class_nums}
        for step, data in enumerate(train_loader):
            images, labels = data
            w_imgs, s_imgs,o_imgs = Variable(images[0]).to(device, non_blocking=True), \
                            Variable(images[1]).to(device, non_blocking=True), \
                            Variable(images[2]).to(device, non_blocking=True)
            labels =  Variable(labels).to(device)
            all_inputs = torch.cat([w_imgs, s_imgs], dim=0)
            all_labels = torch.cat([labels, labels], dim=0)
            bs = w_imgs.shape[0]
            optimizer_warmup.zero_grad()
            # use_gate_mask=torch.tensor([])
            # classes_to_enhance=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
            use_gate_mask = torch.tensor([label in classes_to_enhance for label in all_labels.cpu().numpy()], device=device)
            weighted_feature1, _,pooled_feature,_ =model(all_inputs,use_gate_mask)
            s_logits=weighted_feature1[:bs]
            w_logits=weighted_feature1[bs:2*bs]
            s_f = pooled_feature[:bs]
            w_f = pooled_feature[bs:2*bs]
            entropy_simgs = compute_entropy(s_logits)
            entropy_wimgs = compute_entropy(w_logits)
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_entropy_sum_s[label] += entropy_simgs[i].item()
                class_sample_count_s[label] += 1
                class_entropy_sum_w[label] += entropy_wimgs[i].item()
                class_sample_count_w[label] += 1
                # s_fdi = compute_fdi(s_f,mu_c_s,labels,label)
                # w_fdi = compute_fdi(w_f,mu_c_w,labels,label)
            
            classification_loss = classification_criterion(weighted_feature1[:2*bs],all_labels)
            loss_decoupling = decoupling_loss_fn(pooled_feature, all_labels)
            # div_loss=subregion_diversity_loss(sampling_map1)
            # classification_loss = classification_criterion(weighted_feature1,all_labels)#+classification_criterion(output,labels)
            # 计算对比学习损失
            #center_loss_value = center_loss(pooled_feature, all_labels)
                # 使用统计量计算类内差异

            loss = model.module.compute_total_loss(pooled_feature[:bs], pooled_feature[bs:], weighted_feature1[:2*bs], all_labels)
            
            # classification_loss = classification_criterion(weighted_feature1,all_labels)+classification_criterion(output,labels)
            # _, predicted = torch.max(torch.cat([weighted_feature1,out2],dim=0), 1)
            _, predicted = torch.max(weighted_feature1[:2*bs], 1)
            # 总损失 = 分类损失 + 对比学习损失
            total_loss_value = classification_loss +loss.mean()+loss_decoupling*10#+hard_loss*0.5+div_loss
            total_loss+= classification_loss.item() *all_inputs.size(0)
            total += all_labels.size(0)
            # correct += (predicted == torch.cat([all_labels,all_labels],dim=0)).sum().item()
            correct += (predicted == all_labels).sum().item()
            total_loss_value.backward()
            optimizer_warmup.step()

            

        avg_loss = total_loss / (len(train_loader.dataset))
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        avg_entropy_per_class_s = {label: class_entropy_sum_s[label] / class_sample_count_s[label] 
                                 for label in class_entropy_sum_s.keys()}
        print(f"Average Entropy Per Class_s: {avg_entropy_per_class_s}")
        avg_entropy_per_class_w = {label: class_entropy_sum_w[label] / class_sample_count_w[label] 
                                 for label in class_entropy_sum_w.keys()}
        print(f"Average Entropy Per Class_w: {avg_entropy_per_class_w}")
        min_entropy_s = min(avg_entropy_per_class_s.values())
        max_entropy_s = max(avg_entropy_per_class_s.values())

        min_entropy_w = min(avg_entropy_per_class_w.values())
        max_entropy_w = max(avg_entropy_per_class_w.values())

        normalized_entropy_per_class_s = {
            label: (entropy - min_entropy_s) / (max_entropy_s - min_entropy_s)
            for label, entropy in avg_entropy_per_class_s.items()
        }

        normalized_entropy_per_class_w = {
            label: (entropy - min_entropy_w) / (max_entropy_w - min_entropy_w)
            for label, entropy in avg_entropy_per_class_w.items()
        }

        print(f"Normalized Average Entropy Per Class_s: {normalized_entropy_per_class_s}")
        print(f"Normalized Average Entropy Per Class_w: {normalized_entropy_per_class_w}")

        # 分别计算强增强和弱增强的阈值
        threshold_s = 1.2*sum(avg_entropy_per_class_s.values()) / len(normalized_entropy_per_class_s)
        threshold_w = 1.2*sum(avg_entropy_per_class_w.values()) / len(normalized_entropy_per_class_w)
        
        # 用阈值选择需要增强的类别：强增强和弱增强都需要超过各自的阈值
        if epoch<warmup:
            classes_to_enhance = [
                label for label in avg_entropy_per_class_s.keys()
                if avg_entropy_per_class_s[label] > threshold_s and avg_entropy_per_class_w[label] > threshold_w
            ]
            for label in classes_to_enhance:
                class_frequency[label] += 1
        else:
            classes_to_enhance = [label for label, freq in class_frequency.items() if freq >= 4]
        print(f"Classes to Enhance: {classes_to_enhance}")
        scheduler_warmup.step()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with accuracy: {best_accuracy:.2f}%")
    print(f"Training complete. Best accuracy: {best_accuracy:.2f}%")
    return classes_to_enhance


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

def visualize_cam(img_tensor, sampling_map, fused_sampling_map_down, image_names, save_path="cam_results"):
    """
    可视化 sampling_map 和 fused_sampling_map_down 在输入图像 x 上的 CAM 效果
    :param img_tensor: 输入图像 (B, C, H, W)，例如 (B, 3, 256, 256)
    :param sampling_map: 采样权重图 (B, 1, 8, 8)，需要上采样
    :param fused_sampling_map_down: 融合后的采样权重图 (B, 1, 8, 8)，需要上采样
    :param image_names: 图片的名字 (用于保存文件)
    :param save_path: 保存可视化结果的路径
    """
    os.makedirs(save_path, exist_ok=True)
    
    B, C, H, W = img_tensor.shape  # H=256, W=256
    img_np = img_tensor.cpu().numpy().transpose(0, 2, 3, 1)  # 变成 (B, H, W, C)

    for i in range(B):
        img = img_np[i]  # 取出第 i 张图片
        img = (img - img.min()) / (img.max() - img.min())  # 归一化到 [0,1]
        img = (img * 255).astype(np.uint8)  # 转换为 uint8

        # 确保 img 是 3 通道 (RGB)
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)  # 灰度图转 RGB
        elif img.shape[-1] != 3:
            raise ValueError(f"输入图像的通道数错误: {img.shape[-1]}，请检查输入图像。")

        # 取出第 i 张的权重图 (8x8)
        sampling_map_i = sampling_map[i, 0].cpu().detach().numpy()
        fused_map_i = fused_sampling_map_down[i, 0].cpu().detach().numpy()

        # **上采样到 256x256**
        sampling_map_i = cv2.resize(sampling_map_i, (W, H), interpolation=cv2.INTER_CUBIC)
        fused_map_i = cv2.resize(fused_map_i, (W, H), interpolation=cv2.INTER_CUBIC)

        # **归一化权重图**
        sampling_map_i = cv2.normalize(sampling_map_i, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        fused_map_i = cv2.normalize(fused_map_i, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # **生成热力图**
        heatmap_sampling = cv2.applyColorMap(sampling_map_i, cv2.COLORMAP_JET)
        heatmap_fused = cv2.applyColorMap(fused_map_i, cv2.COLORMAP_JET)

        # **叠加原始图像和热力图**
        overlay_sampling = cv2.addWeighted(img, 0.6, heatmap_sampling, 0.4, 0)
        overlay_fused = cv2.addWeighted(img, 0.6, heatmap_fused, 0.4, 0)

        # **显示结果**
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(img)
        ax[0].set_title("Original Image")
        ax[1].imshow(overlay_sampling)
        ax[1].set_title("Sampling Map CAM")
        ax[2].imshow(overlay_fused)
        ax[2].set_title("Fused Sampling Map CAM")

        # 隐藏坐标轴
        for a in ax:
            a.axis("off")

        plt.tight_layout()

        # **生成文件名**
        base_name = os.path.splitext(image_names[i])[0]  # 去掉扩展名
        save_file = f"{save_path}/{base_name}_cam.png"

        plt.savefig(save_file)
        plt.close()
        
        print(f"✅ Saved CAM visualization: {save_file}")


def get_top_k_classes(logits, k=3):
    """获取每个样本的 top-k 预测类别"""
    _, topk_indices = torch.topk(logits, k, dim=1)
    return topk_indices

def binary_classification_test(binary_model,binary_model1,binary_model2, test_loader, classes_to_enhance, device):
    binary_model.eval()
    binary_model1.eval()
    binary_model2.eval()
    correct = 0
    correct1=0
    correct2=0
    total = 0
    total1=0
    total2=0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            w_imgs, s_imgs,o_imgs = Variable(images[0]).to(device, non_blocking=True), \
                            Variable(images[1]).to(device, non_blocking=True), \
                            Variable(images[2]).to(device, non_blocking=True)
            labels =  Variable(labels).to(device)
            binary_labels1 = (labels.unsqueeze(-1) == torch.tensor(classes_to_enhance, device=device)).any(dim=-1).float().unsqueeze(-1)
            # 前向传播
            
            binary_output_o = binary_model(o_imgs)
            binary_output_o1 = binary_model1(o_imgs)
            binary_output_o2 = binary_model2(o_imgs)
            
            # 计算准确率

            predicted_o = (torch.sigmoid(binary_output_o) > 0.5).float()
            predicted_o1 = (torch.sigmoid(binary_output_o1) > 0.5).float()
            predicted_o2 = (torch.sigmoid(binary_output_o2) > 0.5).float()
            binary_predictions = (torch.sigmoid(binary_output_o) > 0.5).float().squeeze()
            use_gate_mask = binary_predictions.bool()

            binary_predictions1 = (torch.sigmoid(binary_output_o1) > 0.5).float().squeeze()
            use_gate_mask2 = binary_predictions1.bool()
            binary_predictions2 = (torch.sigmoid(binary_output_o2) > 0.5).float().squeeze()
            use_gate_mask3 = binary_predictions2.bool()
            use = use_gate_mask|use_gate_mask2
            total += labels.size(0)
            correct += (predicted_o == binary_labels1).sum().item()
            total1 += labels.size(0)
            correct1 += (predicted_o1 == binary_labels1).sum().item()
            total2 += labels.size(0)
            correct2 += (predicted_o2 == binary_labels1).sum().item()
    
    accuracy = 100 * correct / total
    acc1 = 100*correct1/total1
    acc2 = 100*correct2/total2
    print(f'Test Accuracy:{accuracy :.2f}%,{acc1:.2f}%,{acc2:.2f}%')
    
from sklearn.neighbors import NearestNeighbors
def knn_correct_use_gate_mask(features, use_gate_mask4, k=5):
    """
    使用 KNN 修正 use_gate_mask4（异常样本的二分类结果）
    - features: 样本特征 (N, D)
    - use_gate_mask4: 需要纠正的二分类结果 (N,)
    """
    features_np = features.cpu().numpy()
    mask_np = use_gate_mask4.cpu().numpy().astype(int)  # 转为 0/1

    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(features_np)  # 仅使用测试样本的特征
    _, indices = knn.kneighbors(features_np)  # 查找每个样本的 k 个最近邻索引

    corrected_mask = []
    for i in range(len(mask_np)):
        neighbor_masks = mask_np[indices[i]]  # 获取 k 个邻居的 mask 值
        corrected_value = np.bincount(neighbor_masks).argmax()  # 多数投票
        corrected_mask.append(corrected_value)

    return torch.tensor(corrected_mask, dtype=torch.bool, device=use_gate_mask4.device)

from thop import profile
def evaluate(model,binary_model,binary_model1,binary_model2, test_loader, criterion, classes_to_enhance,device):
    model.eval()  # 设置模型为评估模式
    binary_model.eval()
    binary_model1.eval()
    binary_model2.eval()
    total_loss = 0.0
    correct = 0
    correct_crop = 0
    correct_mix =0
    threshold=0.9
    total = 0
    all_preds=[]
    all_labels=[]
    
    with torch.no_grad():  # 在评估过程中不需要计算梯度
        for images, labels in test_loader:
            w_imgs, s_imgs,o_imgs = Variable(images[0]).to(device, non_blocking=True), \
                            Variable(images[1]).to(device, non_blocking=True), \
                            Variable(images[2]).to(device, non_blocking=True)
            labels =  Variable(labels).to(device)
            all_inputs = torch.cat([w_imgs, s_imgs], dim=0)
            all_labels1 = torch.cat([labels, labels], dim=0)
            classes_to_enhance = torch.tensor([])
            use_gate_mask = torch.tensor([label in classes_to_enhance for label in all_labels1.cpu().numpy()], device=device)
            outputs_cls, _ ,features,_= model(all_inputs,use_gate_mask)
            bs = w_imgs.shape[0]
            s_logits=outputs_cls[:bs]
            w_logits=outputs_cls[bs:2*bs]
            entropy_simgs = compute_entropy(s_logits)
            entropy_wimgs = compute_entropy(w_logits)
            double_thre = (entropy_simgs > threshold) | (entropy_wimgs > threshold)
            use_gate_mask1 =  double_thre.squeeze()
            
            binary_output = binary_model(o_imgs)
            binary_predictions = (torch.sigmoid(binary_output) > 0.5).float().squeeze()
            use_gate_mask = binary_predictions.bool()
            binary_output1 = binary_model1(o_imgs)
            binary_predictions1 = (torch.sigmoid(binary_output1) > 0.5).float().squeeze()
            use_gate_mask2 = binary_predictions1.bool()
            binary_output2 = binary_model2(o_imgs)
            binary_predictions2 = (torch.sigmoid(binary_output2) > 0.5).float().squeeze()
            use_gate_mask3 = binary_predictions2.bool()
            
            
            
            # 前向传播，获取分类输出
           
            combined_masks = torch.stack([use_gate_mask,use_gate_mask1, use_gate_mask2, use_gate_mask3], dim=0)
            use_gate_mask4 = (combined_masks.sum(dim=0) >= 2)
            corrected_use_gate_mask4 = knn_correct_use_gate_mask(features[:bs], use_gate_mask4, k=5)

            classes_to_enhance = torch.tensor([])
            use_gate_mask4 = torch.tensor([label in classes_to_enhance for label in labels.cpu().numpy()], device=device)
            outputs, _ ,_,_= model(o_imgs,corrected_use_gate_mask4)

            # 计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item() * o_imgs.size(0)

            # 计算预测值并统计准确率
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            

    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = 100 * correct / total
    accuracy_crop = 100 * correct_crop / total
    accuracy_mix = 100 * correct_mix / total
    cm = confusion_matrix(all_labels, all_preds)
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    annot = np.where(cm_normalized == 0, "", np.round(cm_normalized, 2))
    # 绘制混淆矩阵
    fig, ax = plt.subplots(figsize=(25, 25),facecolor='white',tight_layout=False)
    sns.heatmap(cm_normalized, annot=annot,fmt="", cmap="Blues", square=True, linewidths=1,linecolor="black", cbar=True,cbar_kws={"shrink": 0.8})

    # 设置标签
    plt.xlabel("Predict label",fontsize=20)
    plt.ylabel("Truth label",fontsize=20)
    plt.title("Confusion Matrix",fontsize=20)
    plt.subplots_adjust(left=0.1,     # 左边距
                   right=0.95,    # 右边距
                   bottom=0.1,    # 下边距
                   top=0.9) 
    plt.savefig("FGSCR-42.png",bbox_inches='tight',  # 裁剪白边
           pad_inches=0.1,       # 保留少量边距
           dpi=300)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    class_recall = cm.diagonal() / cm.sum(axis=0)
    print(class_recall)
    overall_recall = 100*np.mean(class_recall)
    print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%, Test Recall: {overall_recall:.4f}%')
    
    for i, acc in enumerate(class_accuracy):
        print(f'Class {i} Accuracy: {acc:.2f}')


    # print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%')

    return avg_loss, accuracy

def main():
    
    train_loader, test_loader = get_dataloader(batch_size=18, num_workers=4)
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型并移动到设备上
    model = PSAResNet(num_classes=42)
    for param in model.parameters():
        param.requires_grad = True  # 冻结预训练参数
    # **多GPU训练支持**
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)  # 封装多GPU支持
    
    model.to(device)
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    classification_criterion = nn.CrossEntropyLoss()  # 对比学习损失
    # 训练模型

    model1 = PSAResNet(num_classes=42)
    for param in model1.parameters():
        param.requires_grad = True  # 冻结预训练参数
    # **多GPU训练支持**
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model1 = nn.DataParallel(model1)  # 封装多GPU支持
    model1.to(device)
    optimizer_warmup = SGD(model1.parameters(), lr=0.01, momentum=0.9)
    scheduler_warmup = torch.optim.lr_scheduler.StepLR(optimizer_warmup, step_size=10, gamma=0.1)
    classes_to_enhance = warmup_train(model1, train_loader, optimizer_warmup,scheduler_warmup, classification_criterion, device, num_epochs=6,warmup=5)
    # classes_to_enhance=[7,8,17,25,26]
    binary_model1 = BinaryConvNeXt()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        binary_model1 = nn.DataParallel(binary_model1)  # 封装多GPU支持
    binary_model1.to(device)
    optimizer_bin1 = optim.AdamW(binary_model1.parameters(), lr=0.0001, weight_decay=0.05)
    scheduler_bin1 = optim.lr_scheduler.StepLR(optimizer_bin1, step_size=7, gamma=0.1)
    binary_model2 = BinaryEfficientNet()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        binary_model2 = nn.DataParallel(binary_model2)  # 封装多GPU支持
    binary_model2.to(device)
    optimizer_bin2 = optim.AdamW(binary_model2.parameters(), lr=0.0001, weight_decay=0.05)
    scheduler_bin2 = optim.lr_scheduler.StepLR(optimizer_bin2, step_size=7, gamma=0.1)
    binary_model3 = BinaryEfficientNet6()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        binary_model3 = nn.DataParallel(binary_model3)  # 封装多GPU支持
    binary_model3.to(device)
    optimizer_bin3 = optim.AdamW(binary_model3.parameters(), lr=0.0001, weight_decay=0.05)
    scheduler_bin3 = optim.lr_scheduler.StepLR(optimizer_bin3, step_size=7, gamma=0.1)

    
    # 加载最佳模型进行测试
    best_model_path1 = "best_binary_model1_42.pth"
    binary_model1.load_state_dict(torch.load(best_model_path1))
    best_model_path2 = "best_binary_model2_42.pth"
    binary_model2.load_state_dict(torch.load(best_model_path2))
    best_model_path3 = "best_binary_model3_42.pth"
    binary_model3.load_state_dict(torch.load(best_model_path3))
    binary_classification_test(binary_model1,binary_model2,binary_model3, test_loader, classes_to_enhance, device)
    
    best_model_path="best_model_3.pth"
    model.load_state_dict(torch.load(best_model_path))
    evaluate(model,binary_model1,binary_model2,binary_model3, test_loader, classification_criterion,classes_to_enhance, device)


if __name__ == '__main__':
    main() 