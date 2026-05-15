import os

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
import argparse
import regex as re
import torch
from torch import nn
import tensorflow_hub as hub
import numpy as np
import random
import pickle as pkl
import datasets
from torch.utils.data import Dataset, DataLoader, TensorDataset
from collections import Counter
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback)

from sklearn.utils.class_weight import compute_class_weight
import evaluate

RANDOM_STATE = 12345

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed_all(RANDOM_STATE)

TRAIN_VAL_RATIO = 0.8

AUTHOR_EPOCHS = 4

def build_vocab(texts, max_vocab_size=10000):
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())

    vocab = {"<pad>": 0, "<unk>": 1}
    for word, _ in counter.most_common(max_vocab_size - 2):
        vocab[word] = len(vocab)
    return vocab

def tokenize_and_pad(texts, vocab, max_length=200):
    sequences = []
    for text in texts:
        tokens = text.lower().split()
        ids = [vocab.get(t, vocab["<unk>"]) for t in tokens[:max_length]]
        ids = ids + [vocab["<pad>"]] * (max_length - len(ids))
        sequences.append(ids)
    return torch.tensor(sequences, dtype=torch.long)


class LSTMClassifier(nn.Module):
    def __init__(self,vocab_size, embedding_dim, hidden_size, output_size,
            num_layers=1, dropout=0.0, bidirectional=False,padding_idx=0,):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(embedded)

        if self.bidirectional:
            h_final = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_final = h_n[-1]

        return self.fc(h_final)



def train_LSTM(data, models_folder):
    train_df, val_df, mapping = data


    max_vocab_size = 30000
    max_len = 128
    batch_size = 64
    embed_dim = 128
    hidden_dim = 128
    num_epochs = 8
    lr = 1e-3

    vocab = build_vocab(train_df["text"], max_vocab_size)

    x_train = tokenize_and_pad(train_df["text"], vocab, max_len)
    x_val = tokenize_and_pad(val_df["text"], vocab, max_len)

    y_train = torch.tensor(train_df["label"].values, dtype=torch.long)
    y_val = torch.tensor(val_df["label"].values, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size)

    model = LSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=embed_dim,
        hidden_size=hidden_dim,
        output_size=3,
        padding_idx=vocab["<pad>"])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for batch_texts, batch_labels in train_loader:
            optimizer.zero_grad()
        outputs = model(batch_texts)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (predicted == batch_labels).sum().item()
        total += batch_labels.size(0)

        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for batch_texts, batch_labels in test_loader:
                outputs = model(batch_texts)
                _, predicted = outputs.max(1)
                test_correct += (predicted == batch_labels).sum().item()
                test_total += batch_labels.size(0)

        test_acc = test_correct / test_total
        print(f"Epoch {epoch + 1}: Loss={total_loss / len(train_loader):.4f}, "
              f"Train={correct / total:.3f}, Test={test_acc:.3f}")

    model.eval()
    with torch.no_grad():
        _, y_train_pred = model(x_train).max(1)
        _, y_val_pred = model(x_val).max(1)

        made_report(y_train, y_train_pred, y_val, y_val_pred, method_name="LSTM")


    model_backup = {
        "max_len": max_len,
        "model_state_dict": model.state_dict(),
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "vocab": vocab
    }

    model_path = os.path.join(models_folder, "LSTM_model.pkl")
    with open(model_path, "wb") as f:
        pkl.dump(model_backup, f)


def made_report(y_true_train, y_pred_train, y_true_val, y_pred_val, method_name):
    print(f"{method_name}".center(21,"#"))
    print("Train eval".center(21,"-"))
    calc_metrics(y_true_train, y_pred_train)
    print("Validation eval".center(21,"-"))
    calc_metrics(y_true_val, y_pred_val)
    print("\n\n")

def calc_metrics(y_true, y_pred):
    f1_macro = f1_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    s_macro = f"{f1_macro:.5f}".rjust(12," ")
    s_acc = f"{accuracy:.5f}".rjust(12, " ")
    print(f"f1_macro:{s_macro}")
    print(f"accuracy:{s_acc}")

def load_and_transform_data(fpath, ratio=0.8):
    data_raw = pd.read_csv(fpath)
    mapping = {'neutral': 0, 'negative': 1, 'positive': 2}
    data = data_raw.copy()
    data["label"] = data["label"].map(mapping)
    train_df = data.sample(frac=ratio, random_state=RANDOM_STATE)
    val_df = data.drop(train_df.index)
    res = (train_df, val_df, mapping)
    return res


def train_LR_bigrams(data, models_folder=None):
    train_df , val_df , mapping = data
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    x_train = vectorizer.fit_transform(train_df["text"])
    y_train = train_df["label"]

    x_val = vectorizer.transform(val_df["text"])
    y_val = val_df["label"]

    # train
    model = LogisticRegression(max_iter=100, random_state=RANDOM_STATE, solver="saga", l1_ratio=0.5)
    model.fit(x_train, y_train)

    # eval
    y_pred_train = model.predict(x_train)

    y_pred_val = model.predict(x_val)

    made_report(y_train, y_pred_train, y_val, y_pred_val, method_name="LR+Bigrams")

    model_backup = dict(vectorizer=vectorizer, model=model)

    model_name = "LR_bigrams_model.pkl"
    model_path = os.path.join(models_folder, model_name)
    with open(model_path, "wb") as f:
        pkl.dump(model_backup, f)

def train_SVM_trigrams(data, models_folder):
    train_df, val_df, mapping = data
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    x_train = vectorizer.fit_transform(train_df["text"])
    y_train = train_df["label"]

    x_val = vectorizer.transform(val_df["text"])
    y_val = val_df["label"]

    # train
    model = svm.LinearSVC(random_state=RANDOM_STATE, C=0.3, tol=1e-3, penalty="l1")
    model.fit(x_train, y_train)


    y_pred_train = model.predict(x_train)
    y_pred_val = model.predict(x_val)

    made_report(y_train, y_pred_train, y_val, y_pred_val, method_name="SVM+Trigrams")

    model_backup = dict(vectorizer=vectorizer, model=model)

    model_name = "SVM_Trigrams_model.pkl"
    model_path = os.path.join(models_folder, model_name)
    with open(model_path, "wb") as f:
        pkl.dump(model_backup, f)

def train_NaiveBayes(data, models_folder):
    train_df, val_df, mapping = data
    vectorizer = TfidfVectorizer(ngram_range=(1, 1))
    x_train = vectorizer.fit_transform(train_df["text"])
    y_train = train_df["label"]

    x_val = vectorizer.transform(val_df["text"])
    y_val = val_df["label"]

    # train
    model = MultinomialNB()
    model.fit(x_train, y_train)

    y_pred_train = model.predict(x_train)

    y_pred_val = model.predict(x_val)

    made_report(y_train, y_pred_train, y_val, y_pred_val, method_name="NaiveBayes")

    model_backup = dict(vectorizer=vectorizer, model=model)

    model_name = "NB_tdidf_model.pkl"
    model_path = os.path.join(models_folder, model_name)
    with open(model_path, "wb") as f:
        pkl.dump(model_backup, f)

def train_USE_LR(data, models_folder):
    train_df, val_df, mapping = data

    embeding_model = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/tensorFlow2/universal-sentence-encoder/2?tfhub-redirect=true")
    x_train = embeding_model(train_df["text"]).numpy()
    y_train = train_df["label"]

    x_val = embeding_model(val_df["text"]).numpy()
    y_val = val_df["label"]


    model = LogisticRegression(max_iter=300, random_state=RANDOM_STATE, solver="lbfgs")
    model.fit(x_train, y_train)


    y_pred_train = model.predict(x_train)

    y_pred_val = model.predict(x_val)


    made_report(y_train, y_pred_train, y_val, y_pred_val, method_name="USE + LR")

    model_backup = dict(model=model)

    model_name = "USEnLR_model.pkl"
    model_path = os.path.join(models_folder, model_name)
    with open(model_path, "wb") as f:
        pkl.dump(model_backup, f)

def train_authorship_method(data, model_path):
    train_df, val_df, mapping = data
    model_name = "FacebookAI/xlm-roberta-base"
    output_dir = os.path.join(model_path, "author")
    TRAIN_BS = 16
    EVAL_BS = 32
    LR = 1e-5


    label2id = {int(v): int(v) for v in train_df["label"].unique()}
    id2label = {int(v): str(v) for v in train_df["label"].unique()}
    num_labels = 3

    classes = np.array(sorted(train_df["label"].unique()))
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["label"].values
    )

    class_weights = torch.tensor(class_weights, dtype=torch.float)
    class_weights = torch.nn.functional.normalize(class_weights, p=1, dim=0)

    train_dataset = datasets.Dataset.from_pandas(train_df, preserve_index=False)
    val_dataset = datasets.Dataset.from_pandas(val_df, preserve_index=False)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"])

    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_dataset = val_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id={str(k): v for k, v in label2id.items()} if not all(
            isinstance(k, str) for k in label2id) else label2id,
    )

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_metric.compute(predictions=preds, references=labels)["accuracy"],
            "f1_macro": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"],
            "f1_weighted": f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"],
        }

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss


    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=AUTHOR_EPOCHS,
        per_device_train_batch_size=TRAIN_BS,
        per_device_eval_batch_size=EVAL_BS,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_strategy="steps",
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        bf16=False,
        dataloader_num_workers=0,
        seed=RANDOM_STATE,
        remove_unused_columns=False,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )


    train_result = trainer.train()
    print(train_result)

    metrics = trainer.evaluate()
    print("Eval metrics:", metrics)

    trainer.save_model(os.path.join(output_dir, "best_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "best_model"))



def main():
    parser = argparse.ArgumentParser()

    # Optional argument with a default value
    parser.add_argument("-d", "--data_path", type=str, default="./data/raw/RuSentimentsCleanTrain.csv")

    # Boolean flag (True if present, False otherwise)
    parser.add_argument("-o", "--output_folder", type=str, default="./models")

    args = parser.parse_args()
    data_path = args.data_path
    output_folder = args.output_folder

    data_splits = load_and_transform_data(data_path, ratio=TRAIN_VAL_RATIO)
    train_LSTM(data_splits, output_folder)
    train_NaiveBayes(data_splits, output_folder)
    train_LR_bigrams(data_splits, output_folder)
    train_SVM_trigrams(data_splits, output_folder)
    train_USE_LR(data_splits, output_folder)
    train_authorship_method(data_splits, output_folder)



if __name__ == '__main__':
    main()

