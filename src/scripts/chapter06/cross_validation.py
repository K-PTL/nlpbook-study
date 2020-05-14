import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

import logging_dict
from preprocessing import clean_html, tokenize
from stop_watch import stop_watch
from utils import load_dataset

def main():
    logger = logging.getLogger("__name__")

    x, y = load_dataset("data/amazon_reviews_multilingual_JP_v1_00.tsv", n=5000)

    x = [clean_html(text, strip=True) for text in x]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

    vectorizer = TfidfVectorizer(tokenizer=tokenize)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec  = vectorizer.transform(x_test)

    @stop_watch
    def k_fold_cv(clf, x_train, y_train, k_cv=5) -> None:
        logger = logging.getLogger("kfoldcvLigging")
        scores = cross_val_score(clf, x_train, y_train, cv=k_cv)
        logger.debug(f"CV scores (k={k_cv}): {scores}")
        logger.debug("Accuracy: {:.4f} (+/- {:.4f})".format(scores.mean(), scores.std()*2))
        return None

    clf = LogisticRegression(solver="liblinear")
    for k in [3,4,5]:
        k_fold_cv(clf=clf, x_train=x_train_vec, y_train=y_train, k_cv=k)

    clf.fit(x_train_vec, y_train)
    y_pred = clf.predict(x_test_vec)
    score = accuracy_score(y_pred, y_test)
    logger.debug("Accuracy score (test): {:.4f}".format(score))

if __name__=="__main__":
    main()
