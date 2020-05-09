import logging

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

from preprocessing import clean_html, tokenize
from utils import load_dataset, train_and_eval

import logging_dict

def main():
    logger = logging.getLogger('__name__')

    x, y = load_dataset("data/amazon_reviews_multilingual_JP_v1_00.tsv", n=5000)

    logger.debug("●○ Tokenization ○●")
    x = [clean_html(text, strip=True) for text in x]
    x = [" ".join(tokenize(text)) for text in x]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

    logger.debug("●○ Binary ○●")
    vectorizer = CountVectorizer(binary=True)
    train_and_eval(x_train, y_train, x_test, y_test, vectorizer)

    logger.debug("●○ Count ○●")
    vectorizer = CountVectorizer(binary=False)
    train_and_eval(x_train, y_train, x_test, y_test, vectorizer)

    logger.debug("●○ TF-IDF; Uni-gram ○●")
    vectorizer = TfidfVectorizer(ngram_range=(1,1))
    train_and_eval(x_train, y_train, x_test, y_test, vectorizer)

    logger.debug("●○ TF-IDF; Bi-gram ○●")
    vectorizer = TfidfVectorizer(ngram_range=(2,2))
    train_and_eval(x_train, y_train, x_test, y_test, vectorizer)

    logger.debug("●○ TF-IDF; Uni- and Bi-grams ○●")
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    train_and_eval(x_train, y_train, x_test, y_test, vectorizer)
    
    logger.debug("●○ TF-IDF; Uni-, Bi-, and Tri-grams ○●")
    vectorizer = TfidfVectorizer(ngram_range=(1,3))
    train_and_eval(x_train, y_train, x_test, y_test, vectorizer)

    logger.debug("●○ Use MaCab; TF-IDF; Uni-gram ○●") # not written in text
    
    import MeCab
    path_neologd = '/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd'
    t_mecab = MeCab.Tagger("-Owakati && -d {}".format(path_neologd))

    def tokenize_by_mecab(text):
        return list(t_mecab.parse(text).strip().split(" "))

    x, y = load_dataset("data/amazon_reviews_multilingual_JP_v1_00.tsv", n=5000)
    x = [clean_html(text, strip=True) for text in x]
    x = [" ".join(tokenize_by_mecab(text)) for text in x]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)
    vectorizer = TfidfVectorizer(ngram_range=(1,1))
    train_and_eval(x_train, y_train, x_test, y_test, vectorizer)
    
    logger.debug("●○ Use MaCab; TF-IDF; Uni- and Bi-grams ○●") # not written in text
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    train_and_eval(x_train, y_train, x_test, y_test, vectorizer)

    logger.debug("●○ Use MaCab; TF-IDF; Uni-, Bi-, and Tri-grams ○●") # not written in text
    vectorizer = TfidfVectorizer(ngram_range=(1,3))
    train_and_eval(x_train, y_train, x_test, y_test, vectorizer)
    
if __name__=="__main__":
    main()
