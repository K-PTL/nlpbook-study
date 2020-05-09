import logging

from sklearn.model_selection import train_test_split

from preprocess import clean_html, normalize_number, tokenize, tokenize_base_form
from utils import load_dataset, train_and_eval

import logging_dict

def main():
    logger = logging.getLogger('__name__')

    x, y = load_dataset("data/amazon_reviews_multilingual_JP_v1_00.tsv", n=1000)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

    logger.debug("●○ Tokenization only. ○●")
    train_and_eval(x_train, y_train, x_test, y_test, tokenize=tokenize)

    logger.debug("●○ Clean html. ○●")
    train_and_eval(x_train, y_train, x_test, y_test, tokenize=tokenize, preprocessor=clean_html)

    logger.debug("●○ Normalize number. ○●")
    train_and_eval(x_train, y_train, x_test, y_test, tokenize=tokenize, preprocessor=normalize_number)

    logger.debug("●○ Base form. ○●")
    train_and_eval(x_train, y_train, x_test, y_test, tokenize=tokenize_base_form)

    logger.debug("●○ Lower text. ○●")
    train_and_eval(x_train, y_train, x_test, y_test, tokenize=tokenize_base_form, lowercase=True)

    logger.debug("●○ Use MaCab; tokenize only. ○●") # not written in text
    
    import MeCab
    path_neologd = '/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd'
    t_mecab = MeCab.Tagger("-Owakati && -d {}".format(path_neologd))

    def tokenize_by_mecab(text):
        return list(t_mecab.parse(text).strip().split(" "))
    
    train_and_eval(x_train, y_train, x_test, y_test, tokenize=tokenize_by_mecab)

if __name__=="__main__":
    main()
