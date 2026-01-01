import os
import glob
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from scipy.io import loadmat
import json
import matplotlib.pyplot as plt


# -------------------------
# 配置（恢复你原本的目录地址）
# -------------------------
@dataclass
class CFG:
    jpg_dir: str = r"F:\bird--CNN\data.flowers\dataset\jpg"  # 恢复为你原本的目录
    mat_dir: str = r"F:\bird--CNN\data.flowers\dataset"  # 恢复为你原本的目录
    test_images_dir: str = r"F:\bird--CNN\data.flowers\test"  # 恢复为你原本的目录

    num_classes: int = 102
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 0  # Windows下设为0避免卡顿
    lr: float = 3e-4
    epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择GPU/CPU
    cat_to_name_path: str = r"F:\bird--CNN\data.flowers\dataset\cat_to_name.json"  # 恢复为你原本的目录

    save_path: str = "flowers102_resnet50_best.pt"


# -------------------------
# 数据集：读取 .mat 标签与划分（逻辑不变）
# -------------------------
class Flowers102MatDataset(Dataset):
    """
    Oxford Flowers 102:
      - 图片: image_00001.jpg ... image_08189.jpg (共8189张)
      - 标签: imagelabels.mat -> labels, 1..102
      - 划分: setid.mat -> trnid, valid, tstid (都是图片编号，1-based)
    """

    def __init__(self, jpg_dir, mat_dir, split: str, transform=None):
        super().__init__()
        self.jpg_dir = jpg_dir
        self.mat_dir = mat_dir
        self.transform = transform

        imagelabels_path = os.path.join(mat_dir, "imagelabels.mat")
        setid_path = os.path.join(mat_dir, "setid.mat")

        if not os.path.isfile(imagelabels_path):
            raise FileNotFoundError(f"找不到 {imagelabels_path}。请把 imagelabels.mat 放到 mat_dir 目录下。")
        if not os.path.isfile(setid_path):
            raise FileNotFoundError(f"找不到 {setid_path}。请把 setid.mat 放到 mat_dir 目录下。")

        labels_mat = loadmat(imagelabels_path)
        setid_mat = loadmat(setid_path)

        # labels: shape (1, 8189) 或 (8189, 1)
        labels = labels_mat["labels"].squeeze()  # -> (8189,)
        # split ids are 1-based indices
        if split == "train":
            ids = setid_mat["trnid"].squeeze()
        elif split == "val":
            ids = setid_mat["valid"].squeeze()
        elif split == "test":
            ids = setid_mat["tstid"].squeeze()
        else:
            raise ValueError("split 必须是 'train' / 'val' / 'test'")

        self.samples = []
        for img_id in ids:
            img_id_int = int(img_id)
            img_name = f"image_{img_id_int:05d}.jpg"
            img_path = os.path.join(jpg_dir, img_name)
            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"找不到图片: {img_path}（请确认 jpg_dir 正确）")
            label_1based = int(labels[img_id_int - 1])
            label_0based = label_1based - 1
            self.samples.append((img_path, label_0based))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, y = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, y


# -------------------------
# 工具函数（逻辑不变）
# -------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)  # 确保数据在GPU上
        logits = model(x)
        loss = ce(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return loss_sum / max(total, 1), correct / max(total, 1)


def train():
    cfg = CFG()

    print("Device:", cfg.device)
    print("jpg_dir:", cfg.jpg_dir)
    print("mat_dir:", cfg.mat_dir)

    # 数据增强
    train_tfms = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 加载数据集
    train_ds = Flowers102MatDataset(cfg.jpg_dir, cfg.mat_dir, split="train", transform=train_tfms)
    val_ds = Flowers102MatDataset(cfg.jpg_dir, cfg.mat_dir, split="val", transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

    # 模型：预训练ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, cfg.num_classes)
    model = model.to(cfg.device)  # 模型移到GPU

    # 优化器与调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    ce = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        correct, total = 0, 0

        for x, y in train_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()

        scheduler.step()

        # 计算指标
        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, cfg.device)

        dt = time.time() - t0
        print(f"Epoch {epoch:02d}/{cfg.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {dt:.1f}s")

        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(), "cfg": cfg.__dict__}, cfg.save_path)
            print(f"  ✅ Save best -> {cfg.save_path} (val_acc={best_val_acc:.4f})")

    print("Training done. Best val_acc =", best_val_acc)


def load_cat_to_name(json_path: str):
    """
    读取 Oxford102 类别映射：返回 {int: (英文名称, 中文名称)}
    JSON结构：{"1": ["pink primrose", "粉报春"], ...}
    """
    if not os.path.isfile(json_path):
        raise FileNotFoundError(
            f"找不到类别映射文件: {json_path}\n"
            "请确保 cat_to_name.json 存在且路径正确。"
        )

    with open(json_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    # 转换为 {int: (英文, 中文)} 格式
    cat_map = {}
    for k, v in m.items():
        try:
            key = int(k)
            # 容错：确保是列表且有2个元素
            if isinstance(v, list) and len(v) >= 2:
                en_name = v[0].strip()
                cn_name = v[1].strip()
                cat_map[key] = (en_name, cn_name)
            else:
                # 结构错误时用默认值
                cat_map[key] = (f"class_{key}", f"类别_{key}")
        except (ValueError, IndexError):
            # 键转换失败或索引错误
            continue

    return cat_map


def show_prediction(img_pil, en_name: str, cn_name: str, conf: float, topk_en, topk_cn, topk_probs):
    """
    可视化预测结果：同时显示英文+中文
    """
    # 设置Matplotlib支持中文
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]  # 黑体 + 英文备用
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    # 1) 显示图片 + 中英名称+置信度
    plt.figure(figsize=(8, 6))
    plt.imshow(img_pil)
    plt.axis("off")
    plt.title(f"{cn_name} ({en_name})  \n置信度: {conf:.4f}", fontsize=14)

    # 2) Top-5 概率条形图（显示中英名称）
    plt.figure(figsize=(10, 5))
    # 组合中英名称作为x轴标签
    topk_labels = [f"{cn}\n({en})" for en, cn in zip(topk_en, topk_cn)]
    plt.bar(range(len(topk_probs)), topk_probs, color="#4CAF50")
    plt.xticks(range(len(topk_labels)), topk_labels, rotation=0, ha="center", fontsize=10)
    plt.ylabel("置信度", fontsize=12)
    plt.title("Top-5 预测结果", fontsize=14)
    plt.ylim(0, 1.05)  # y轴范围0-1.05
    # 给每个柱子标注数值
    for i, prob in enumerate(topk_probs):
        plt.text(i, prob + 0.02, f"{prob:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.show()


# -------------------------
# 推理：识别 test 文件夹下的图片（逻辑不变）
# -------------------------
@torch.no_grad()
def infer_folder():
    cfg = CFG()

    # 检查模型权重
    if not os.path.isfile(cfg.save_path):
        raise FileNotFoundError(f"找不到模型权重 {cfg.save_path}，请先运行train()训练模型。")

    # 加载模型
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, cfg.num_classes)

    ckpt = torch.load(cfg.save_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(cfg.device)
    model.eval()

    # 图片预处理
    tfm = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 读取测试图片
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(cfg.test_images_dir, e)))

    if not paths:
        print(f"在 {cfg.test_images_dir} 未找到图片（支持格式：{exts}）。")
        return

    # 加载类别映射（英文+中文）
    cat_to_name = load_cat_to_name(cfg.cat_to_name_path)
    if not cat_to_name:
        print("警告：类别映射文件加载失败，仅显示类别编号。")

    # 批量推理
    for p in sorted(paths):
        try:
            img = Image.open(p).convert("RGB")
            x = tfm(img).unsqueeze(0).to(cfg.device)

            # 模型预测
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[0]  # 概率分布
            pred0 = int(prob.argmax().item())  # 0-based
            pred1 = pred0 + 1  # 1-based
            conf = float(prob[pred0].item())

            # 获取中英名称
            if pred1 in cat_to_name:
                en_name, cn_name = cat_to_name[pred1]
            else:
                en_name = f"class_{pred1}"
                cn_name = f"类别_{pred1}"

            # Top-5 预测结果
            topk = 5
            top_probs, top_idx = torch.topk(prob, k=topk)
            top_probs = top_probs.detach().cpu().numpy().tolist()
            top_idx = top_idx.detach().cpu().numpy().tolist()  # 0-based
            top_ids = [i + 1 for i in top_idx]  # 1-based

            # 获取Top-5的中英名称
            topk_en = []
            topk_cn = []
            for idx in top_ids:
                if idx in cat_to_name:
                    te, tc = cat_to_name[idx]
                else:
                    te = f"class_{idx}"
                    tc = f"类别_{idx}"
                topk_en.append(te)
                topk_cn.append(tc)

            # 打印结果
            print(f"\n{os.path.basename(p)}:")
            print(f"  预测结果：{cn_name} ({en_name})")
            print(f"  类别编号：{pred1}")
            print(f"  置信度：{conf:.4f}")
            print(f"  Top-5 预测：")
            for i, (te, tc, tp) in enumerate(zip(topk_en, topk_cn, top_probs)):
                print(f"    {i + 1}. {tc} ({te}) - {tp:.4f}")

            # 可视化
            show_prediction(img, en_name, cn_name, conf, topk_en, topk_cn, top_probs)

        except Exception as e:
            print(f"处理图片 {p} 时出错：{str(e)}")
            continue


if __name__ == "__main__":
    # 先训练模型，再执行推理
    # 若已训练过，可注释掉train()，直接运行infer_folder()
    train()
    infer_folder()