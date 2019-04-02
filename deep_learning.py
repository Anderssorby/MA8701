import os
from argparse import ArgumentParser

import matplotlib
import numpy as np
import pandas as pd
from keras.callbacks import ProgbarLogger
from keras.layers import Dense, LSTM, Dropout, Embedding, Activation, TimeDistributed
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

if os.getenv("DISPLAY") is None:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

# T is the length of data you want to run right now (the full dataset takes forever
# mdf is for setting the min_df value in the TfidfVectorizer function (google is good)
# -- when building the vocabulary for back of words ignore terms that have a document
# frequency strictly lower than min_df

T = 30000
mdf = 50
VALIDATION_SPLIT = 0.25
MAX_NB_WORDS = 1000
MAX_SEQUENCE_LENGTH = 1000
GLOVE_DIR = "180"


def load_data(word_embedding):
    # get data from csv files
    data = pd.read_csv('train.csv', usecols=['description', 'deal_probability'])
    desc = (data['description'])
    y = (data['deal_probability'])
    del data

    if word_embedding:
        return word_embeddings(desc, y)

    # break up data into train and test data
    train_desc, test_desc, train_y, test_y = train_test_split(desc, y, test_size=VALIDATION_SPLIT, random_state=23)

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


def word_embeddings(desc, labels):
    texts = desc  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    # labels = []  # list of label ids

    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]

    return x_train, y_train, x_val, y_val

def embedding_layer(word_index):
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    return embedding_layer



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


def create_simple_model(input_shape=1836):
    model = Sequential(name="dense")

    model.add(Dense(100, input_dim=input_shape, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    return model


def create_lstm_model(hidden_size=1836, use_dropout=True):
    model = Sequential(name="LSTM")

    model.add(Embedding())
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(LSTM(hidden_size, return_sequences=True))
    if use_dropout:
        model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(vocabulary)))
    model.add(Activation('softmax'))


def plot_model_history(history):
    plt.figure()

    # plt.plot(history['binary_accuracy'])

    plt.plot(history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.figure()
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()


def train(train_x, train_y, model: Sequential, epochs,  **kwargs):
    progbar = ProgbarLogger()

    history = model.fit(
        x=train_x,
        y=train_y,
        epochs=epochs,
        callbacks=[progbar]
    )

    plot_model_history(history.history)


def main(lr, model_filename, **kwargs):
    train_x, train_y, test_x, test_y = load_data(
        word_embedding=kwargs["word_embedding"]
    )
    opt = Adam(lr=lr)

    model = create_simple_model(train_x.shape[1])
    model.compile(
        optimizer=opt,
        loss="mean_squared_error",
        metrics=['acc'],
    )
    model.summary()

    train(train_x, train_y, model, **kwargs)

    #score = model.test_on_batch(x=test_x, y=test_y)

    #print("Test score:", score)

    model.save(model_filename, overwrite=False)


if __name__ == "__main__":
    ap = ArgumentParser()

    hyp = ap.add_argument_group("hyper_parameters")
    hyp.add_argument('--lr', help="learning rate", type=int, default=0.05)
    ap.add_argument('--epochs', help="Number of training epochs", type=int, default=10)

    ap.add_argument('--batch_size', '-b', help="The batch size", type=int, default=100)
    ap.add_argument('--word-embedding', dest="word_embedding", type=bool, default=False)
    ap.add_argument('--model-filename', '-f', dest="model_filename", type=str, default="deep_model.h5")

    kwargs = vars(ap.parse_args())
    main(**kwargs)

