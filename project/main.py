"""
入口文件：只负责流程编排，不含任何模型/训练细节。

训练流程：
  1. warmup_train     → 识别难分类类别 classes_to_enhance
  2. train_binary_head → 在 warmup 模型上训练二分类头
  3. contrastive_train → 正式对比训练主分类模型
  4. train_binary_head → 在更强的主模型上再训练二分类头
  5. evaluate          → 最终测试与混淆矩阵
"""

import os
import torch
import torch.nn as nn
from torch.optim import SGD
import torch.optim as optim

from dataloader import get_dataloader
from models    import PSAResNet
from train     import warmup_train, contrastive_train, train_binary_head
from evaluate  import evaluate, binary_classification_test

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'


def build_model(num_classes, device):
    model = PSAResNet(num_classes=num_classes)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    return model.to(device)


def main():
    # ── 基础设置 ───────────────────────────────────────────────
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 23
    train_loader, test_loader = get_dataloader(batch_size=12, num_workers=4)
    cls_criterion = nn.CrossEntropyLoss()
 
    # ── Step 1：Warmup，识别难分类类别 ────────────────────────────
    model_warmup = build_model(num_classes, device)
    opt_w = SGD(model_warmup.parameters(), lr=0.01, momentum=0.9)
    sch_w = optim.lr_scheduler.StepLR(opt_w, step_size=10, gamma=0.1)
 
    classes_to_enhance = warmup_train(
        model_warmup, train_loader, opt_w, sch_w,
        cls_criterion, device,
        num_epochs=6, warmup=5,
        save_path=CKPT_WARMUP,
    )
    # classes_to_enhance = [5,11,13,19,20]
    print(f"\n>>> Hard classes: {classes_to_enhance}\n")
 
    # ── Step 2：正式对比训练，得到最优分类权重 ───────────────────
    model_cls = build_model(num_classes, device)
    opt_m = SGD(model_cls.parameters(), lr=0.01, momentum=0.9)
    sch_m = optim.lr_scheduler.StepLR(opt_m, step_size=10, gamma=0.1)
 

 
    # ── Step 3：复制一份模型，专门训练 binary_head ───────────────
    # 从最优分类权重出发，保证 binary_head 在好特征上学习
    model_binary = build_model(num_classes, device)
    model_binary.load_state_dict(torch.load(CKPT_CLS, map_location=device))
    print(f"Copied cls weights → model_binary (base: {CKPT_CLS})")
 

    model_binary.load_state_dict(torch.load(CKPT_FINAL, map_location=device))
    binary_classification_test(model_binary, test_loader, classes_to_enhance, device)
 
    # ── Step 4：两个模型各加载各自最优权重，送入 evaluate ────────
    model_cls.load_state_dict(torch.load(CKPT_CLS, map_location=device))
    # model_binary.load_state_dict(torch.load(CKPT_FINAL, map_location=device))
    print(f"model_cls    ← {CKPT_CLS}")
    print(f"model_binary ← {CKPT_FINAL}")
 
    evaluate(
        model_cls, model_binary,
        test_loader, cls_criterion,
        classes_to_enhance, device,
        save_path="confusion_matrix_final.png",
    )

if __name__ == '__main__':
    main()