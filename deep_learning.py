import os
from argparse import ArgumentParser

import pandas as pd
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ProgbarLogger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib
if os.getenv("DISPLAY") is None:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


# T is the length of data you want to run right now (the full dataset takes forever
# mdf is for setting the min_df value in the TfidfVectorizer function (google is good)
# -- when building the vocabulary for back of words ignore terms that have a document
# frequency strictly lower than min_df

T = 30000
mdf = 50


def load_data():
    # get data from csv files
    data = pd.read_csv('train.csv', usecols=['description', 'deal_probability'])
    desc = (data['description'])
    y = (data['deal_probability'])
    del data

    # break up data into train and test data
    train_desc, test_desc, train_y, test_y = train_test_split(desc, y, test_size=0.25, random_state=23)

    # shrink training data to T
    train_desc = train_desc[:T]
    train_y = train_y[:T]

    # Replace nans with spaces
    train_desc.fillna(" ", inplace=True)
    test_desc.fillna(" ", inplace=True)

    bow_file = "bow.npy"
    if not os.path.isfile(bow_file):
        # word embedding
        train_x, test_x = bag_of_words(train_desc, test_desc)

        np.save(bow_file, (train_x, test_x))
    else:
        train_x, test_x = np.load(bow_file)

    return train_x, train_y, test_x, test_y


def bag_of_words(train_desc, test_desc):
    # Get "bag of words" transformation of the data -- see example in Lasso book discussed in class
    # also: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction

    vec = TfidfVectorizer(ngram_range=(1, 1),
                          min_df=mdf,
                          max_df=0.9,
                          lowercase=True,
                          strip_accents='unicode',
                          sublinear_tf=True)

    train_x = vec.fit_transform(train_desc)
    test_x = vec.transform(test_desc)

    return train_x, test_x


def create_simple_model():
    model = Sequential(name="dense")
    input_shape = 1836

    model.add(Dense(100, input_dim=input_shape, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    return model


def plot_model_history(history):
    plt.figure()

    # plt.plot(history['binary_accuracy'])

    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.figure()
    plt.plot(history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()


def train(model: Sequential, epochs, **kwargs):
    train_x, train_y, test_x, test_y = load_data()

    progbar = ProgbarLogger()

    history = model.fit(
        x=train_x,
        y=train_y,
        epochs=epochs,
        callbacks=[progbar]
    )

    model.test_on_batch(x=test_x, y=test_y)

    plot_model_history(history.history)


def main(lr, model_filename, **kwargs):
    opt = Adam(lr=lr)

    model = create_simple_model()
    model.compile(
        optimizer=opt,
        loss="mean_squared_error",
        metrics=['acc'],
    )
    model.summary()

    train(model, **kwargs)

    model.save(model_filename, overwrite=False)


if __name__ == "__main__":
    ap = ArgumentParser()

    hyp = ap.add_argument_group("hyper_parameters")
    hyp.add_argument('--lr', help="learning rate", type=int, default=0.05)
    ap.add_argument('--epochs', help="Number of training epochs", type=int, default=100)

    ap.add_argument('--model-filename', '-f', dest="model_filename", type=str, default="deep_model.h5")

    kwargs = vars(ap.parse_args())
    main(**kwargs)
