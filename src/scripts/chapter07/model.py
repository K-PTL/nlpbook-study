import logging 

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, Sequential

import logging_dict

logger = logging.getLogger("modelLogging")

def create_model(vocab_size, label_size, hidden_size=16):
    model = Sequential()
    model.add( Dense(hidden_size, activation="relu", input_shape=(vocab_size,)) )
    model.add( Dense(label_size,  activation="softmax") )
    logger.debug(model.summary())
    return model
