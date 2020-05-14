import logging

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

import logging_dict
from model import create_model
from preprocessing import clean_html, tokenize
from stop_watch import stop_watch
from utils import load_dataset, plot_history

def main():
    logger = logging.getLogger("__name__")

    # load dataset
    x, y = load_dataset("data/amazon_reviews_multilingual_JP_v1_00.tsv", n=5000)

    # feature engineering
    x = [clean_html(text, strip=True) for text in x]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

    # vectorization
    vectorizer = CountVectorizer(tokenizer=tokenize)
    x_train = vectorizer.fit_transform(x_train)
    x_test  = vectorizer.transform(x_test)
    x_train = x_train.toarray()
    x_test  = x_test.toarray()

    # setting hyperparameters
    vocab_size = len(vectorizer.vocabulary_)
    label_size = len(set(y_train))

    # build model
    model = create_model(vocab_size, label_size)
    model.compile(loss="sparse_categorical_crossentropy", 
                  optimizer="adam",
                  metrics=["accuracy"])

    # declare callbacks
    filepath = "./log/model.h5"
    
    cb_early = EarlyStopping(
        monitor="val_loss",        # monitored parameter, default: val_loss (validation loss)
        min_delta=0,               # default: 0
        patience=3,                # epochs for stopping, default: 0
        verbose=0,                 # default: 0
        mode="auto",               # default: auto
        baseline=None,             # default: None
        restore_best_weights=False # default: False
    )

    cb_modelcp = ModelCheckpoint(
        filepath=filepath,         # path for save model, default: nothing; NECCESARY ARGUMENT
        monitor="val_loss",        # monitored parameter, default: val_loss (validation loss)
        verbose=0,                 # default: 0
        save_best_only=True,       # if True, save best model only (save HDD usage), default: False
        save_weights_only=False,   # if True, save weights parameter only (not include model architecture), default: False
        mode="auto",               # default: False
        period=1                   # default: False
    )

    # when use Tensorboard, run commands on terminal as shown below;
    #   ``` tensorboard --logdir=./logs (this is default) --bind_all ```
    # after training model, u can access to http://localhost:6006 and use tensorboard.
    # if u wanna exit tensorboard, press Ctrl+C on terminal.
    cb_tensorb = TensorBoard(
        log_dir="./log",             # path for save params plotting on tensorboard, delfault: ./logs
        histogram_freq=0,            # default: 0
        batch_size=32,               # default: 32
        write_graph=True,            # default: True
        write_grads=False,           # default: False
        write_images=False,          # default: False
        embeddings_freq=0,           # default: 0
        embeddings_layer_names=None, # default: None
        embeddings_metadata=None,    # default: None
        embeddings_data=None,        # default: None
        update_freq="epoch"          # default: epoch
    )

    """callbacks = [
        cb_early,
        cb_modelcp,
        cb_tensorb
    ]"""

    callbacks = [
        EarlyStopping(
            patience=3,                # epochs for stopping, default: 0
        ),
        ModelCheckpoint(
            filepath=filepath,         # path for save model, default: nothing; NECCESARY ARGUMENT
            save_best_only=True,       # if True, save best model only (save HDD usage), default: False
        ),
        TensorBoard(
            log_dir="./log",             # path for save params plotting on tensorboard, delfault: ./logs
        )
    ]
    

    # training model
    #@stop_watch
    def train_model(x_train, y_train):
        return model.fit(x_train, y_train,
                         validation_split=0.2,
                         epochs=100,
                         batch_size=32,
                         callbacks=callbacks)
    
    history = train_model(x_train, y_train)

    # load saved model
    model = load_model(filepath)

    # describe model
    # if fail to run this, try command on terminal;
    #    pip install pydot==1.2.3 pydot_ng && apt-get install graphviz
    plot_model(model, to_file="./log/model.png")

    # predict by model
    text = "このアプリ超最高！"
    vec  = vectorizer.transform([text])
    y_pred = model.predict(vec.toarray())
    logger.debug(f"""input text is "{text}".""")
    logger.debug("predict: ", y_pred)

    # plot acc and loss graphs
    plot_history(history)

if __name__=="__main__":
    main()
