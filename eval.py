import os
from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import argparse
import regex as re
import torch
from torch import nn
import tensorflow_hub as hub
import numpy as np
import pickle as pkl
import datasets
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments)
import evaluate

RANDOM_STATE = 12345
TRAIN_VAL_RATIO = 0.8

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

def calc_metrics(y_true, y_pred):
    f1_macro = f1_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    s_macro = f"{f1_macro:.5f}".rjust(12," ")
    s_acc = f"{accuracy:.5f}".rjust(12, " ")
    print(f"f1_macro:{s_macro}")
    print(f"accuracy:{s_acc}")

def made_report_test(y_true_test, y_pred_test, method_name):
    print(f"{method_name}".center(21,"#"))
    print("Test eval".center(21,"-"))
    calc_metrics(y_true_test, y_pred_test)
    print("\n")

def load_test_data(fpath):
    mapping = {'neutral':0, 'negative':1, 'positive':2}
    data_raw = pd.read_csv(fpath)
    data = data_raw.copy()
    data["label"] = data["label"].map(mapping)
    res = (data, mapping)
    return res



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



def eval_LSTM(test_df, models_folder):

    model_path = os.path.join(models_folder, "LSTM_model.pkl")
    with open(model_path, "rb") as f:
        model_backup = pkl.load(f)


    max_len = model_backup['max_len']
    model_state_dict = model_backup["model_state_dict"]
    embed_dim = model_backup["embed_dim"]
    hidden_dim = model_backup["hidden_dim"]
    vocab = model_backup["vocab"]


    model = LSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=embed_dim,
        hidden_size=hidden_dim,
        output_size=3,
        padding_idx=vocab["<pad>"])

    model.load_state_dict(model_state_dict)

    x_test = tokenize_and_pad(test_df["text"], vocab, max_len)
    y_test = torch.tensor(test_df["label"].values, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        _, y_test_pred = model(x_test).max(1)

    made_report_test(y_test, y_test_pred, "LSTM")

    return y_test_pred


def eval_LR_bigrams(test_df, models_folder):
    model_name = "LR_bigrams_model.pkl"
    model_path = os.path.join(models_folder, model_name)

    with open(model_path, "rb") as f:
        model_backup = pkl.load(f)

    vectorizer = model_backup["vectorizer"]
    model = model_backup["model"]

    x_test = vectorizer.transform(test_df["text"])
    y_test = test_df["label"]

    y_pred = model.predict(x_test)

    made_report_test(y_test, y_pred, "LR+Bigrams")

    return y_pred


def eval_SVM_trigrams(test_df, models_folder):
    model_name = "SVM_Trigrams_model.pkl"
    model_path = os.path.join(models_folder, model_name)

    with open(model_path, "rb") as f:
        model_backup = pkl.load(f)

    vectorizer = model_backup["vectorizer"]
    model = model_backup["model"]

    x_test = vectorizer.transform(test_df["text"])
    y_test = test_df["label"]

    y_pred = model.predict(x_test)

    made_report_test(y_test, y_pred, "SVM+Trigrams")

    return y_pred


def eval_NaiveBayes(test_df, models_folder):
    model_name = "NB_tdidf_model.pkl"
    model_path = os.path.join(models_folder, model_name)

    with open(model_path, "rb") as f:
        model_backup = pkl.load(f)

    vectorizer = model_backup["vectorizer"]
    model = model_backup["model"]

    x_test = vectorizer.transform(test_df["text"])
    y_test = test_df["label"]

    y_pred = model.predict(x_test)

    made_report_test(y_test, y_pred, "NaiveBayes")

    return y_pred

def eval_USE_LR(test_df, models_folder):
    model_name = "USEnLR_model.pkl"
    model_path = os.path.join(models_folder, model_name)

    with open(model_path, "rb") as f:
        model_backup = pkl.load(f)

    model = model_backup["model"]

    embeding_model = hub.load(
        "https://www.kaggle.com/models/google/universal-sentence-encoder/tensorFlow2/universal-sentence-encoder/2?tfhub-redirect=true"
    )

    x_test = embeding_model(test_df["text"]).numpy()
    y_test = test_df["label"]

    y_pred = model.predict(x_test)

    made_report_test(y_test, y_pred, "USE + LR")

    return y_pred

def eval_authorship_method(test_df, models_folder):
    model_dir = os.path.join(models_folder, "author", "best_model")

    test_dataset = datasets.Dataset.from_pandas(test_df, preserve_index=False)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    def tokenize_fn(batch):
        return tokenizer(batch["text"])

    test_dataset = test_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)

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

    eval_args = TrainingArguments(
        output_dir=os.path.join(models_folder, "author", "eval_tmp"),
        per_device_eval_batch_size=32,
        dataloader_num_workers=0,
        fp16=torch.cuda.is_available(),
        bf16=False,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    pred_output = trainer.predict(test_dataset)

    y_test = pred_output.label_ids
    y_pred = pred_output.predictions.argmax(axis=-1)

    print("Authorship method\n")
    print("Test eval")
    made_report_test(y_test, y_pred, "Authorship method")

    return y_pred

def main():
    parser = argparse.ArgumentParser()

    # Optional argument with a default value
    parser.add_argument("-d", "--data_path", type=str, default="./data/raw/RuSentimentsCleanTest.csv")

    # Boolean flag (True if present, False otherwise)
    parser.add_argument("-m", "--model_path", type=str, default="models")

    args = parser.parse_args()
    data_path = args.data_path
    model_path = args.model_path

    data_test, mapping = load_test_data(data_path)
    eval_USE_LR(data_test, model_path)
    eval_LSTM(data_test, model_path)
    eval_LR_bigrams(data_test, model_path)
    eval_USE_LR(data_test, model_path)
    eval_NaiveBayes(data_test, model_path)
    eval_authorship_method(data_test, model_path)



if __name__ == '__main__':
    main()
