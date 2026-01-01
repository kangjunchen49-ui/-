import os
import re
import numpy as np
import pandas as pd
import torch
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding, EarlyStoppingCallback
import datasets
import nlpaug.augmenter.word as naw
import nltk

# ================= 配置区域 =================
warnings.filterwarnings('ignore')

# 修复：静默 NLTK 下载，防止刷屏
nltk.data.path.append(os.path.expanduser('~/nltk_data'))
for resource in ['averaged_perceptron_tagger', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


class Config:
    DATA_PATH = "music_lyrics_genre_bilingual_full.csv"
    LYRICS_FOLDER = r"F:\bird--CNN\data.music\test"
    MODEL_SAVE_PATH = "./music_genre_model_pro"

    MODEL_NAME = "xlm-roberta-base"

    MAX_LENGTH = 256
    BATCH_SIZE = 8
    EPOCHS = 8
    LEARNING_RATE = 2e-5

    LABEL_SMOOTHING = 0.1

    GENRE_LIST = ["流行", "摇滚", "嘻哈", "民谣", "爵士", "电子", "乡村", "金属", "R&B", "古典"]


# ================= 1. 高级数据增强工具 =================

# 初始化增强器 (同义词替换)
try:
    aug_en = naw.SynonymAug(aug_src='wordnet', lang='eng', aug_p=0.3)
except:
    print("Warning: nlpaug 初始化失败，将跳过同义词增强")
    aug_en = None


def clean_text(text):
    if pd.isna(text): return ""
    text = str(text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def augment_text(text, genre):
    if aug_en is None: return text
    is_english = len(re.findall(r'[a-zA-Z]', text)) > len(re.findall(r'[\u4e00-\u9fa5]', text))

    if is_english:
        try:
            augmented_text = aug_en.augment(text)
            return augmented_text[0] if isinstance(augmented_text, list) else augmented_text
        except:
            return text
    else:
        return text


def process_data(data_path, tokenizer):
    print(f"正在读取数据集: {data_path}")
    full_df = pd.read_csv(data_path, encoding="utf-8")
    full_df = full_df.dropna(subset=["lyrics", "genre"])
    full_df = full_df[full_df["genre"].isin(Config.GENRE_LIST)]

    le = LabelEncoder()
    le.fit(Config.GENRE_LIST)

    train_df_raw, test_df_raw = train_test_split(
        full_df, test_size=0.15, stratify=full_df["genre"], random_state=42
    )

    train_data = []
    print("正在进行训练集增强 (滑动窗口 + 同义词替换)...")

    window_size = 200
    step_size = 100

    for _, row in tqdm(train_df_raw.iterrows(), total=len(train_df_raw)):
        text = clean_text(row["lyrics"])
        label = le.transform([row["genre"]])[0]
        words = text.split()

        segments = []
        if len(words) <= window_size:
            segments.append(text)
        else:
            for i in range(0, len(words) - 50, step_size):
                segments.append(" ".join(words[i: i + window_size]))

        for seg in segments:
            train_data.append({"text": seg, "label": label})
            if len(seg) > 50:
                aug_seg = augment_text(seg, row["genre"])
                if aug_seg != seg:
                    train_data.append({"text": aug_seg, "label": label})

    test_data = []
    for _, row in test_df_raw.iterrows():
        text = clean_text(row["lyrics"])
        label = le.transform([row["genre"]])[0]
        words = text.split()
        if len(words) <= window_size:
            test_data.append({"text": text, "label": label})
        else:
            mid = len(words) // 2
            start = max(0, mid - window_size // 2)
            segment = " ".join(words[start: start + window_size])
            test_data.append({"text": segment, "label": label})

    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    print(f"📊 增强后统计 | 训练样本: {len(train_df)} | 测试样本: {len(test_df)}")

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df["label"]),
        y=train_df["label"]
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    train_ds = datasets.Dataset.from_pandas(train_df).class_encode_column("label")
    test_ds = datasets.Dataset.from_pandas(test_df).class_encode_column("label")

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=Config.MAX_LENGTH)

    return train_ds.map(tokenize, batched=True), test_ds.map(tokenize, batched=True), le, class_weights


# ================= 2. 自定义 Trainer (已修复报错) =================

class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None

    # 🔥 修复点：添加 num_items_in_batch 参数以兼容新版 Transformers
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


# ================= 3. 训练流程 =================

def train_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    train_ds, test_ds, le, class_weights = process_data(Config.DATA_PATH, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME, num_labels=len(le.classes_)
    )

    print("⚖️ 类别权重计算结果:", class_weights)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = classification_report(labels, predictions, output_dict=True)['weighted avg'].values()
        acc = (predictions == labels).mean()
        return {"accuracy": acc, "f1": f1}

    training_args = TrainingArguments(
        output_dir="./music_model_pro",
        learning_rate=Config.LEARNING_RATE,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        num_train_epochs=Config.EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        label_smoothing_factor=Config.LABEL_SMOOTHING,
        use_cpu=not torch.cuda.is_available(),
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("🚀 开始高精度训练...")
    trainer.train()

    print(f"💾 保存最终模型到: {Config.MODEL_SAVE_PATH}")
    trainer.save_model(Config.MODEL_SAVE_PATH)
    tokenizer.save_pretrained(Config.MODEL_SAVE_PATH)
    np.save(os.path.join(Config.MODEL_SAVE_PATH, "classes.npy"), le.classes_)

    return trainer, test_ds, le


# ================= 4. 预测与评估 =================

def evaluate_and_plot(trainer, test_ds, le):
    preds = np.argmax(trainer.predict(test_ds).predictions, axis=-1)
    labels = test_ds["label"]

    print("\n" + classification_report(labels, preds, target_names=le.classes_))

    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_matrix(labels, preds), annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix (Pro Model)")
    if not os.path.exists(Config.MODEL_SAVE_PATH): os.makedirs(Config.MODEL_SAVE_PATH)
    plt.savefig(os.path.join(Config.MODEL_SAVE_PATH, "confusion_matrix_pro.png"))
    print("✅ 评估完成")


def predict_folder(folder_path):
    print(f"\n🔮 开始预测文件夹: {folder_path}")
    if not os.path.exists(Config.MODEL_SAVE_PATH): return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_SAVE_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(Config.MODEL_SAVE_PATH).to(device)
    classes = np.load(os.path.join(Config.MODEL_SAVE_PATH, "classes.npy"), allow_pickle=True)
    model.eval()

    results = []
    files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

    for filename in tqdm(files):
        path = os.path.join(folder_path, filename)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
        except:
            continue

        clean = clean_text(text)
        if len(clean) < 10: continue

        words = clean.split()
        chunks = [" ".join(words[i:i + 256]) for i in range(0, len(words), 200)]
        if not chunks: chunks = [clean]

        chunk_probs = []
        for chunk in chunks:
            inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=256, padding=True).to(device)
            with torch.no_grad():
                probs = torch.nn.functional.softmax(model(**inputs).logits, dim=-1)
                chunk_probs.append(probs.cpu().numpy()[0])

        avg_probs = np.mean(chunk_probs, axis=0)
        top_idx = np.argmax(avg_probs)

        results.append({
            "filename": filename,
            "genre": classes[top_idx],
            "confidence": avg_probs[top_idx],
            "text_snippet": clean[:50]
        })

    pd.DataFrame(results).to_csv("final_results_pro.csv", index=False, encoding='utf-8-sig')
    print("✅ 预测结果已保存至 final_results_pro.csv")


if __name__ == "__main__":
    trainer, test_ds, le = train_pipeline()
    evaluate_and_plot(trainer, test_ds, le)
    predict_folder(Config.LYRICS_FOLDER)
