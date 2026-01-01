import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from sklearn.model_selection import train_test_split

# ===================== 1. 配置参数 =====================
DATA_ROOT = r"F:\bird--CNN\data.bird\dataset"
TEST_IMG_DIR = r"F:\bird--CNN\data.bird\test"
MODEL_PATH = "bird_efficientnetb4.pth"
BATCH_SIZE = 16  # EfficientNetB4参数量大，减小batch size避免OOM
EPOCHS = 25
LEARNING_RATE = 5e-5  # 更小的学习率适配EfficientNet
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")


# ===================== 2. 数据集预处理 =====================
def load_bird_labels():
    # 读取类别名称文件
    classes_file = os.path.join(DATA_ROOT, "classes.txt")
    idx2name = {}
    with open(classes_file, "r", encoding="utf-8") as f:
        for line in f:
            idx, name = line.strip().split(" ", 1)
            idx2name[int(idx) - 1] = name.replace("_", " ")

    # 读取图片路径和标签
    images_file = os.path.join(DATA_ROOT, "images.txt")
    labels_file = os.path.join(DATA_ROOT, "image_class_labels.txt")

    img_paths = []
    img_labels = []
    with open(images_file, "r") as f:
        for line in f:
            _, path = line.strip().split(" ", 1)
            img_paths.append(os.path.join(DATA_ROOT, "images", path))

    with open(labels_file, "r") as f:
        for line in f:
            _, label = line.strip().split(" ")
            img_labels.append(int(label) - 1)

    return idx2name, img_paths, img_labels


class BirdDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            print(f"图片加载失败: {img_path}")
            return None, None
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# 适配EfficientNetB4的预处理（输入尺寸380×380）
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.RandomResizedCrop(380, scale=(0.7, 1.0)),  # EfficientNetB4推荐输入380×380
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomAffine(degrees=20, translate=(0.15, 0.15), shear=10),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.RandomGrayscale(p=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


# ===================== 3. 构建EfficientNetB4模型 =====================
def build_model(num_classes=200):
    # 加载预训练的EfficientNetB4
    model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)

    # 解冻后1/3的层，平衡拟合能力和泛化能力
    for name, param in model.named_parameters():
        if "features.6" in name or "features.7" in name or "classifier" in name:
            param.requires_grad = True  # 解冻高层特征和分类头
        else:
            param.requires_grad = False

    # 替换分类头（适配CUB_200_2011的200类）
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),  # 增加Dropout防过拟合
        nn.Linear(in_features, num_classes)
    )

    model = model.to(DEVICE)
    return model


# ===================== 4. 训练函数（带早停+学习率衰减） =====================
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    best_acc = 0.0
    train_losses = []
    val_accs = []
    patience = 5  # 早停耐心值
    patience_counter = 0

    # 余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        valid_samples = 0
        for images, labels in train_loader:
            if images is None:
                continue
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            valid_samples += images.size(0)

        epoch_loss = running_loss / valid_samples if valid_samples > 0 else 0
        train_losses.append(epoch_loss)
        scheduler.step()  # 更新学习率

        # 验证阶段
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                if images is None:
                    continue
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_acc = correct / total if total > 0 else 0
        val_accs.append(epoch_acc)

        # 保存最优模型+早停逻辑
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), MODEL_PATH)
            patience_counter = 0
            print(f"更新最优模型，当前最优精度：{best_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停触发！Epoch: {epoch + 1}, 最优精度：{best_acc:.4f}")
                break

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Val Acc: {epoch_acc:.4f}")

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss")

    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label="Validation Accuracy", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Validation Accuracy")
    plt.tight_layout()
    plt.savefig("efficientnet_training_curve.png")
    plt.show()

    return model


# ===================== 5. 预测+可视化函数 =====================
def predict_bird(model, img_path, idx2name, transform):
    image = Image.open(img_path).convert("RGB")
    img_original = image.copy()
    image = transform(image).unsqueeze(0)
    image = image.to(DEVICE)

    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        top5_prob, top5_idx = torch.topk(probabilities, 5)

    results = []
    for i in range(5):
        class_idx = top5_idx[0][i].item()
        prob = top5_prob[0][i].item()
        bird_name = idx2name[class_idx]
        results.append((bird_name, prob))

    return img_original, results


def visualize_prediction(img_original, results, img_name):
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 解决中文乱码
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(12, 6))
    # 显示测试图片
    plt.subplot(1, 2, 1)
    plt.imshow(img_original)
    plt.axis("off")
    plt.title("测试图片", fontsize=12)

    # 显示Top5预测结果
    plt.subplot(1, 2, 2)
    bird_names = [r[0] for r in results]
    probs = [r[1] for r in results]
    y_pos = np.arange(len(bird_names))

    bars = plt.barh(y_pos, probs, align="center", color="#1f77b4")
    plt.yticks(y_pos, bird_names, fontsize=10)
    plt.xlabel("概率", fontsize=11)
    plt.title("Top5预测结果", fontsize=12)
    plt.xlim(0, 1.05)

    # 标注概率值
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{prob:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"efficientnet_prediction_{img_name}.png", dpi=150, bbox_inches="tight")
    plt.show()


# ===================== 6. 主程序 =====================
if __name__ == "__main__":
    # 1. 加载标签和数据
    idx2name, all_img_paths, all_labels = load_bird_labels()

    # 2. 划分训练/验证集
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_img_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )

    # 3. 数据加载器
    train_transform, val_transform = get_transforms()
    train_dataset = BirdDataset(train_paths, train_labels, train_transform)
    val_dataset = BirdDataset(val_paths, val_labels, val_transform)


    # 过滤无效样本
    def collate_fn(batch):
        batch = [x for x in batch if x[0] is not None]
        return torch.utils.data.dataloader.default_collate(batch)


    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)

    # 4. 构建模型
    model = build_model(num_classes=200)

    # 5. 训练/加载模型
    if not os.path.exists(MODEL_PATH):
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  # AdamW优化器
        model = train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)
    else:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("加载已训练的EfficientNetB4模型成功！")

    # 6. 测试集预测
    test_img_files = [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    for img_file in test_img_files:
        img_path = os.path.join(TEST_IMG_DIR, img_file)
        print(f"\n识别图片: {img_file}")

        # 预测
        img_original, results = predict_bird(model, img_path, idx2name, val_transform)

        # 打印结果
        print("Top 5 预测结果:")
        for i, (bird_name, prob) in enumerate(results, 1):
            print(f"{i}. {bird_name} - {prob:.4f}")

        # 可视化
        visualize_prediction(img_original, results, os.path.splitext(img_file)[0])