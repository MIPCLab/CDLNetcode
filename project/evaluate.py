import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors

from utils import compute_entropy


# ──────────────────────────────────────────────────────────────
# KNN 投票修正
# ──────────────────────────────────────────────────────────────

def knn_correct_use_gate_mask(features, use_gate_mask, k=5):
    features_np = features.cpu().numpy()
    mask_np = use_gate_mask.cpu().numpy().astype(int)
    n_samples = len(features_np)
    # 样本数不足时直接返回原 mask，无需 KNN 修正
    if n_samples <= 1:
        return use_gate_mask.clone()
    # k 不能超过样本数
    k = min(k, n_samples)
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(features_np)
    _, indices = knn.kneighbors(features_np)
    corrected = [np.bincount(mask_np[indices[i]]).argmax() for i in range(n_samples)]
    return torch.tensor(corrected, dtype=torch.bool, device=use_gate_mask.device)


# ──────────────────────────────────────────────────────────────
# 二分类测试（仅 binary_head）
# ──────────────────────────────────────────────────────────────

def binary_classification_test(model, test_loader, classes_to_enhance, device):
    """
    用 model.binary_head 的完整 forward 输出做二分类测试，
    统一走 raw_model（绕过 DataParallel scatter）。
    """
    model.eval()
    raw_model      = model.module if hasattr(model, 'module') else model
    classes_tensor = torch.tensor(classes_to_enhance, device=device)
    correct = total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data[:2]
            o_imgs  = Variable(images[2]).to(device)
            labels  = Variable(labels).to(device)
            binary_labels = (
                (labels.unsqueeze(-1) == classes_tensor).any(dim=-1)
                .float().unsqueeze(-1)
            )
            gate   = torch.zeros(o_imgs.size(0), dtype=torch.bool, device=device)
            _, _, _, _, binary_logit = raw_model(o_imgs, gate)

            predicted = (torch.sigmoid(binary_logit) > 0.5).float()
            correct  += (predicted == binary_labels).sum().item()
            total    += labels.size(0)

    print(f"[BinaryHead] Test accuracy: {100 * correct / total:.2f}%")


# ──────────────────────────────────────────────────────────────
# 完整评估
# ──────────────────────────────────────────────────────────────
def evaluate(model_cls, model_binary,
             test_loader, criterion,
             classes_to_enhance, device,
             save_path="confusion_matrix.png",
             entropy_threshold=0.9, knn_k=5):
    """
    model_cls    ← ckpt_cls.pth    负责最终多分类推理
    model_binary ← ckpt_final.pth  负责生成 binary gate
 
    推理流程：
      1. model_cls  + 弱/强增强图 → entropy gate
      2. model_binary + 原图      → binary gate
      3. 取并集 → KNN 修正
      4. model_cls + 原图 + 修正 gate → 最终分类
    """
    model_cls.eval()
    model_binary.eval()
    raw_cls    = model_cls.module    if hasattr(model_cls,    "module") else model_cls
    raw_binary = model_binary.module if hasattr(model_binary, "module") else model_binary
 
    total_loss = correct = total = 0
    all_preds, all_labels_list = [], []
 
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[:2]
            w_imgs = Variable(images[0]).to(device, non_blocking=True)
            s_imgs = Variable(images[1]).to(device, non_blocking=True)
            o_imgs = Variable(images[2]).to(device, non_blocking=True)
            labels = Variable(labels).to(device)
            bs = w_imgs.shape[0]
 
            # ── Step 1：entropy gate（用 model_cls）────────────────
            all_aug   = torch.cat([w_imgs, s_imgs], dim=0)
            empty_2bs = torch.zeros(all_aug.size(0), dtype=torch.bool, device=device)
            logits_aug, _, feats_aug, _, _ = raw_cls(all_aug, empty_2bs)
 
            s_ent = compute_entropy(logits_aug[:bs])
            w_ent = compute_entropy(logits_aug[bs:])
            entropy_gate = (
                (s_ent > entropy_threshold) | (w_ent > entropy_threshold)
            ).view(-1).bool()
 
            # ── Step 2：binary gate（用 model_binary）──────────────
            empty_bs = torch.zeros(bs, dtype=torch.bool, device=device)
            _, _, _, _, binary_logit = raw_binary(o_imgs, empty_bs)
            binary_gate = (torch.sigmoid(binary_logit) > 0.5).view(-1).bool()
 
            # ── Step 3：合并 + KNN 修正 ────────────────────────────
            # combined_gate  = entropy_gate | binary_gate
            combined_gate  = binary_gate
            corrected_gate = knn_correct_use_gate_mask(feats_aug[:bs], combined_gate, k=knn_k)
 
            # ── Step 4：最终分类（用 model_cls）────────────────────
            outputs, _, _, _, _ = raw_cls(o_imgs, corrected_gate)
 
            loss        = criterion(outputs, labels)
            total_loss += loss.item() * bs
            _, predicted = torch.max(outputs, 1)
            total   += bs
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels_list.extend(labels.cpu().numpy())
 
    # ── 指标 ──────────────────────────────────────────────────
    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = 100 * correct / total
 
    cm      = confusion_matrix(all_labels_list, all_preds)
    row_sum = cm.sum(axis=1).astype("float")
    col_sum = cm.sum(axis=0).astype("float")
 
    cm_normalized   = np.where(row_sum[:, None] > 0,
                               cm / row_sum[:, None], 0.0)
    class_recall    = np.where(row_sum > 0, cm.diagonal() / row_sum, 0.0)
    class_precision = np.where(col_sum > 0, cm.diagonal() / col_sum, 0.0)
 
    print(f"Test  loss={avg_loss:.4f}  acc={accuracy:.2f}%  "
          f"recall={100*np.mean(class_recall):.2f}%  "
          f"precision={100*np.mean(class_precision):.2f}%")
    for i in range(len(class_recall)):
        print(f"  Class {i:2d}  recall={class_recall[i]:.3f}  "
              f"precision={class_precision[i]:.3f}")
 
    # ── 混淆矩阵 ─────────────────────────────────────────────
    annot = np.where(cm_normalized == 0, "", np.round(cm_normalized, 2))
    fig, ax = plt.subplots(figsize=(25, 25), facecolor="white")
    sns.heatmap(cm_normalized, annot=annot, fmt="", cmap="Blues",
                square=True, linewidths=1, linecolor="black", cbar=True)
    plt.xlabel("Predicted label", fontsize=20)
    plt.ylabel("True label",      fontsize=20)
    plt.title("Confusion Matrix", fontsize=20)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.close()
    print(f"Confusion matrix → {save_path}")
 
    return avg_loss, accuracy