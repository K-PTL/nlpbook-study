import logging
import string
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
