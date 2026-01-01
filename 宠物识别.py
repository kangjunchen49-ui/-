import os
import glob
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

import matplotlib.pyplot as plt


# -------------------------
# 配置
# -------------------------
@dataclass
class CFG:
    images_dir: str = r"F:\bird--CNN\data.pets\images"
    ann_dir: str = r"F:\bird--CNN\data.pets\annotations"
    test_images_dir: str = r"F:\bird--CNN\data.pets\test"

    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    lr: float = 3e-4
    epochs: int = 10

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "pets_resnet50_best.pt"


# -------------------------
# 读取 annotations/*.txt
# -------------------------
def read_split_file(path: str):
    """
    读取 trainval.txt / test.txt
    每行一般: <img_stem> <class_id> <species_id> <breed_id?>
    我们只用前两列：img_stem, class_id(1-based)
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到划分文件: {path}")

    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            img_stem = parts[0]
            class_id = int(parts[1])  # 1-based
            items.append((img_stem, class_id))
    return items


def build_class_map(list_txt_path: str):
    """
    从 annotations/list.txt 构建 class_id -> class_name（品种名）
    list.txt 里通常包含所有样本行，第二列是 class_id
    我们通过 class_id 收集一个代表性的品种名（从 img_stem 推断）。
    """
    if not os.path.isfile(list_txt_path):
        # 没有 list.txt 也能训练/推理，只是无法显示花名/品种名
        return None

    id_to_name = {}
    with open(list_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            img_stem = parts[0]
            class_id = int(parts[1])
            # img_stem 形如 "Abyssinian_1"，去掉末尾 _数字
            # 品种名里本身有下划线，如 "American_Bulldog"
            breed = "_".join(img_stem.split("_")[:-1]) or img_stem
            if class_id not in id_to_name:
                id_to_name[class_id] = breed
    return id_to_name


# -------------------------
# Dataset
# -------------------------
class PetsClsDataset(Dataset):
    """
    Oxford-IIIT Pets 分类数据集
    - images: xxx.jpg
    - annotations: trainval.txt / test.txt / list.txt
    标签：class_id（1..37，通常是37个品种）
    """
    def __init__(self, images_dir, split_items, transform=None):
        super().__init__()
        self.images_dir = images_dir
        self.transform = transform

        self.samples = []
        for img_stem, class_id_1based in split_items:
            # 官方图片是 .jpg（几乎都是）
            img_path = os.path.join(images_dir, img_stem + ".jpg")
            if not os.path.isfile(img_path):
                # 兜底：如果有别的扩展名，尝试 glob
                cand = glob.glob(os.path.join(images_dir, img_stem + ".*"))
                if cand:
                    img_path = cand[0]
                else:
                    raise FileNotFoundError(f"找不到图片: {img_stem} 对应文件（检查 images_dir）")

            y = class_id_1based - 1  # 0-based
            self.samples.append((img_path, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, y


# -------------------------
# 评估
# -------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return loss_sum / max(total, 1), correct / max(total, 1)


# -------------------------
# 可视化
# -------------------------
def show_prediction(img_pil, title_text: str, topk_names, topk_probs):
    plt.figure()
    plt.imshow(img_pil)
    plt.axis("off")
    plt.title(title_text)

    plt.figure()
    plt.bar(range(len(topk_probs)), topk_probs)
    plt.xticks(range(len(topk_names)), topk_names, rotation=45, ha="right")
    plt.title("Top-5 probabilities")
    plt.tight_layout()
    plt.show()


# -------------------------
# 训练
# -------------------------
def train():
    cfg = CFG()
    print("Device:", cfg.device)

    train_items = read_split_file(os.path.join(cfg.ann_dir, "trainval.txt"))
    val_items = read_split_file(os.path.join(cfg.ann_dir, "test.txt"))  # 简单起见：用官方 test 当验证
    # 如果你想严格：可以从 trainval 再切一部分做 val

    # 类别数：从 trainval/test 的 class_id 最大值推断
    num_classes = max([cid for _, cid in train_items + val_items])
    print("num_classes =", num_classes)

    train_tfms = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_ds = PetsClsDataset(cfg.images_dir, train_items, transform=train_tfms)
    val_ds = PetsClsDataset(cfg.images_dir, val_items, transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    ce = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss, correct, total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)  # ✅ 数据上 GPU
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            optimizer.step()

            run_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()

        scheduler.step()

        train_loss = run_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, cfg.device)

        dt = time.time() - t0
        print(f"Epoch {epoch:02d}/{cfg.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {dt:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "num_classes": num_classes,
            }, cfg.save_path)
            print(f"  ✅ Save best -> {cfg.save_path} (val_acc={best_val_acc:.4f})")

    print("Training done. Best val_acc =", best_val_acc)


# -------------------------
# 推理 + 可视化（test 目录）
# -------------------------
@torch.no_grad()
def infer_folder():
    cfg = CFG()

    if not os.path.isfile(cfg.save_path):
        raise FileNotFoundError(f"找不到模型权重 {cfg.save_path}，请先训练。")

    # 类别名映射（可选，但强烈建议有）
    id_to_name = build_class_map(os.path.join(cfg.ann_dir, "list.txt"))

    ckpt = torch.load(cfg.save_path, map_location=cfg.device)
    num_classes = int(ckpt["num_classes"])

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(cfg.device)  # ✅ 模型上 GPU
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(cfg.test_images_dir, e)))

    if not paths:
        print(f"在 {cfg.test_images_dir} 没找到图片（支持 {exts}）。")
        return

    for p in sorted(paths):
        img = Image.open(p).convert("RGB")
        x = tfm(img).unsqueeze(0).to(cfg.device)
        logits = model(x)

        prob = torch.softmax(logits, dim=1)[0]          # (num_classes,)
        pred0 = int(prob.argmax().item())               # 0-based
        pred1 = pred0 + 1                               # 1-based for mapping
        conf = float(prob[pred0].item())

        if id_to_name:
            pred_name = id_to_name.get(pred1, f"class_{pred1}")
        else:
            pred_name = f"class_{pred1}"

        # Top-5
        topk = min(5, num_classes)
        top_probs, top_idx = torch.topk(prob, k=topk)
        top_probs = top_probs.detach().cpu().numpy().tolist()
        top_idx = top_idx.detach().cpu().numpy().tolist()
        top_ids = [i + 1 for i in top_idx]
        top_names = [(id_to_name.get(i, f"class_{i}") if id_to_name else f"class_{i}") for i in top_ids]

        print(f"{os.path.basename(p)} -> {pred_name} (class_id={pred1})  confidence={conf:.4f}")

        show_prediction(
            img,
            title_text=f"{pred_name}  (conf={conf:.4f})",
            topk_names=top_names,
            topk_probs=top_probs
        )


if __name__ == "__main__":
    train()
    infer_folder()
