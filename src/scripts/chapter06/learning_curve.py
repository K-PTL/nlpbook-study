import logging

from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import logging_dict
from preprocessing import clean_html, tokenize
from utils import load_dataset, plot_learning_curve

def main():
    logger = logging.getLogger("__name__")
    SPLITS = 5
    SEED   = 44

    x, y = load_dataset("data/amazon_reviews_multilingual_JP_v1_00.tsv", n=5000)

    x = [clean_html(text, strip=True) for text in x]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=SEED)

    vectorizer = TfidfVectorizer(tokenizer=tokenize)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec  = vectorizer.transform(x_test)

    cv  = ShuffleSplit(n_splits=SPLITS, test_size=0.2, random_state=SEED)
    clf = LogisticRegression(solver="liblinear")
    title = "Learning Curves"
    save_path = f"./log/n-{SPLITS}_learning_curve.png"
    plot_learning_curve(clf, title, x_train_vec, y_train, cv=cv, save_path=save_path)
    logger.debug(f"Saving learning curve has done as {save_path}.")

if __name__=="__main__":
    main()
