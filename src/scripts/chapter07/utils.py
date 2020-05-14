import logging

import numpy as np
import string
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve

import logging_dict
from stop_watch import stop_watch


def filter_by_ascii_rate(text, threshold=0.9):
    """
        テキスト中の文字のうち、英語の比率が閾値 threshold 以下だったら日本語だろうということ
    """
    ascii_letters = set(string.printable)
    # ascii_letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    rate = sum(c in ascii_letters for c in text)/len(text)
    return rate <= threshold

def load_dataset(filename, n=5_000, state=6):
    logger = logging.getLogger("loadLogging")
    df = pd.read_csv(filename, delimiter="\t")
    logger.debug(f"shape of '{filename}'; {df.shape}")
    logger.debug(df.head())

    # ここの処理;ラベルのマッピングはchapter04ではやっていない
    # converts multi(1,2,3,4,5)-class to binary(0,1)-class
    mapping = {1:0, 2:0, 4:1, 5:1}
    df = df[df.star_rating != 3]
    df.star_rating = df.star_rating.map(mapping)

    # extracts Japanese texts
    is_JP = df.review_body.apply(filter_by_ascii_rate)
    df = df[is_JP]

    # sampling
    df = df.sample(frac=1, random_state=state) # shuffle
    grouped = df.groupby("star_rating") # grouping by each labels
    df = grouped.head(n=n) # pick out samples equally from each labels
    return df.review_body.values, df.star_rating.values

def plot_history(history):
    import matplotlib.pyplot as plt
    
    logger = logging.getLogger("plotHistLogging")
    # Setting
    logger.debug("history.history.keys() are; ", history.history.keys())
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    #acc = history.history["acc"]
    #val_acc = history.history["val_acc"]
    acc = history.history["accuracy"] # in textbook, key==acc but in actual key:acc don't exist.
    val_acc = history.history["val_accuracy"] # val_acc is as same as acc; truely key is val_accuracy.
    epochs = range(1, len(loss)+1)

    # Plotting loss
    fig, axes = plt.subplots(2,1,figsize=(10,10))
    datas  = [[loss, val_loss], [acc, val_acc]]
    labels = ["Loss", "Accuracy"]
    for ax, data, label in zip(axes, datas, labels):
        ax.plot(epochs, data[0], "r", label=f"Training {label.lower()}")
        ax.plot(epochs, data[1], "b", label=f"Validation {label.lower()}")
        ax.set_title(f"Training and Validation {label.lower()}")
        ax.set_xlabel("Epochs")
        ax.set_ylabel(label)
        ax.legend()
    plt.tight_layout()
    plt.savefig("./log/history.png")
    plt.show()
