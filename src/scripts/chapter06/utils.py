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

@stop_watch
def train_and_eval(x_train, y_train, x_test, y_test, vectorizer) -> None:
    """
    vectorizer(sklearn.feature_extraction.text) :  OneHot, Count, or tf-idf encoders are able to be chosen.
    """
    logger = logging.getLogger('trnevLogging')
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec  = vectorizer.transform(x_test)
    clf = LogisticRegression(solver="liblinear")
    clf.fit(x_train_vec, y_train)
    y_pred = clf.predict(x_test_vec)
    score = accuracy_score(y_test, y_pred)
    logger.debug("{:.4f}".format(score))

@stop_watch
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, 
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), save_path=None):
    import matplotlib.pyplot as plt # avoid catching too many DEBUG massages when not use plt method
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve( # test_score = CV score
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std  = np.std( train_scores, axis=1)
    test_scores_mean  = np.mean(test_scores,  axis=1)
    test_scores_std   = np.std( test_scores,  axis=1)
    
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean-train_scores_std, train_scores_mean+train_scores_std,
                     alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean-test_scores_std,   test_scores_mean+test_scores_std,
                     alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Traininig score")
    plt.plot(train_sizes,  test_scores_mean,  "o-", color="g", label="Cross-validation score")

    plt.legend(loc="best")
    plt.tight_layout()
    if save_path is not None: plt.savefig(save_path)
    plt.show()
