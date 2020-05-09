import logging

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import logging_dict
from preprocessing import clean_html, tokenize
from utils import load_dataset

def main():
    logger = logging.getLogger('__name__')

    import MeCab
    path_neologd = '/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd'
    t_mecab = MeCab.Tagger("-Owakati && -d {}".format(path_neologd))
    def tokenize_by_mecab(text):
        return list(t_mecab.parse(text).strip().split(" "))
    
    use_tokens = [tokenize, tokenize_by_mecab]
    t_names = ["janome", "MeCab"]

    logger.debug("Loading dataset...")
    x, y = load_dataset("data/amazon_reviews_multilingual_JP_v1_00.tsv", n=5000)
    x = [clean_html(text, strip=True) for text in x]
    
    for use_token, t_name in zip(use_tokens, t_names):
        logger.debug("●○ {} ○●".format(t_name))
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

        logger.debug("Count vectorizing...")
        vectorizer = CountVectorizer(tokenizer=use_token)
        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(x_test)
        logger.debug(f"x_train's shape is {x_train.shape}")
        logger.debug(f" x_test's shape is {x_test.shape}")

        logger.debug("Selecting features...")
        selector = SelectKBest(k=7000, score_func=mutual_info_classif)
        # k; number of used features, mutal_info_classif; 相互情報量
        selector.fit(x_train, y_train)
        x_train_new = selector.transform(x_train)
        x_test_new  = selector.transform(x_test)
        logger.debug(f"x_train_new's shape is {x_train_new.shape}")
        logger.debug(f" x_test_new's shape is {x_test_new.shape}")

        logger.debug("Evaluating...")
        clf = LogisticRegression(solver="liblinear")
        clf.fit(x_train_new, y_train)
        y_pred = clf.predict(x_test_new)
        score = accuracy_score(y_test, y_pred)
        logger.debug("{:.4f}".format(score))

if __name__=="__main__":
    main()
