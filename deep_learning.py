import os
from argparse import ArgumentParser

import matplotlib
import numpy as np
import pandas as pd
import pickle
import yaml
from keras.callbacks import ProgbarLogger
from keras.layers import Dense, LSTM, Dropout, Embedding, Activation, TimeDistributed
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import keras.backend as K

if os.getenv("DISPLAY") is None:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_data(word_embedding, filename='train.csv'):
    # get data from csv files
    data = pd.read_csv(filename,
                       usecols=['description', 'deal_probability', 'region', 'city', 'parent_category_name',
                                'category_name', 'price', 'param_1', 'param_2', 'param_3', 'title'])
    print("Loaded data", data.shape)
    desc = data['description']
    # desc = pd.DataFrame([data['title'], data['description'], data['param_1'], data['param_2'], data['param_3']]).transpose()
    region = pd.get_dummies(pd.Categorical(data['region']), prefix='category')
    city = pd.get_dummies(pd.Categorical(data['city']), prefix='category')
    y = (data['deal_probability'])
    data['desc_length'] = data['description'].str.len()
    del data

    if word_embedding:
        word_index = word_embeddings(desc, y)

    desc.fillna(" ", inplace=True)

    # break up data into train and test data
    train_desc, test_desc, train_y, test_y = train_test_split(desc, y, test_size=VALIDATION_SPLIT, random_state=23)


    # shrink training data to T
    train_desc = train_desc[:sub_set]
    train_y = train_y[:sub_set]
    print(train_desc.shape)

    # Replace nans with spaces
    # [t.fillna(" ", inplace=True) for t in train_desc.flatten()]

    bow_file = "bow-%d" % sub_set
    if not os.path.isfile(bow_file):
        vec = bag_of_words()

        train_x = vec.fit_transform(train_desc)
        test_x = vec.transform(test_desc)

        pickler = pickle.Pickler(open(bow_file, "wb"))
        pickler.dump(vec)
        print("Pickled bow file", train_x.shape, bow_file)
        # np.save(bow_file, vec)
    else:
        # vec = np.load(bow_file)
        pickler = pickle.Unpickler(open(bow_file, 'rb'))
        vec = pickler.load()

        train_x = vec.transform(train_desc)
        test_x = vec.transform(test_desc)

    return train_x, train_y, test_x, test_y


def word_embeddings(texts, labels):
    labels_index = {}  # dictionary mapping label name to numeric id
    # labels = []  # list of label ids

    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    # labels = to_categorical(np.asarray(labels))
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
    f = open(os.path.join(GLOVE_DIR, 'model.txt'))
    for line in f:
        values = line.split()
        w = values[0].split("_")
        word = w[0]
        # wclass = w[1]
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


def bag_of_words():
    # Get "bag of words" transformation of the data -- see example in Lasso book discussed in class
    # also: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
    print("mdf=", mdf)
    vec = TfidfVectorizer(ngram_range=(1, 1),
                          min_df=mdf,
                          max_df=0.9,
                          lowercase=True,
                          strip_accents='unicode',
                          sublinear_tf=True)

    return vec


def create_simple_model(input_shape, units_list=None, **kwargs):
    model = Sequential(name="dense")

    if not units_list:
        units_list = [1000, 100]

    for units in units_list:
        model.add(Dense(units=units, input_dim=input_shape, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    return model


def create_lstm_model(word_index, hidden_size, use_dropout=True, **kwargs):
    model = Sequential(name="LSTM")

    model.add(embedding_layer(word_index))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(LSTM(hidden_size, return_sequences=True))
    if use_dropout:
        model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(vocabulary)))
    model.add(Activation('softmax'))


models = {
    "simple": create_simple_model,
    "lstm": create_lstm_model
}


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
    plt.savefig(figure_filename)


def train(train_x, train_y, test_x, test_y, model, epochs):
    progbar = ProgbarLogger()

    history = model.fit(
        x=train_x,
        y=train_y,
        epochs=epochs,
        callbacks=[progbar],
        #validation_freq=1,
        validation_data=(test_x, test_y)
    )

    plot_model_history(history.history)


def kaggle_test(model_filename):
    model = load_model(model_filename)
    
    filename = 'test.csv'
    data = pd.read_csv(filename,
                       usecols=['description', 'region', 'city', 'parent_category_name', 'item_id',
                                'category_name', 'price', 'param_1', 'param_2', 'param_3', 'title'])
    desc = data['description']
    # desc = pd.DataFrame([data['title'], data['description'], data['param_1'], data['param_2'], data['param_3']]).transpose()
    global sub_set
    if not sub_set:
        sub_set = 80000

    bow_file = "bow-%d" % sub_set

    vec = pickle.load(open(bow_file, 'rb'))

    test_x = vec.fit_transform(desc)
    result = model.predict(test_x)

    dframe = pd.DataFrame(data={'item_id': data['item_id'], 'deal_probability': result})
    dframe.to_csv(model_filename+".csv", index=False)


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))


def main(lr, model_filename, **kwargs):
    train_x, train_y, test_x, test_y = load_data(
        word_embedding=kwargs["word_embedding"]
    )
    opt = Adam(lr=lr)
    model_builder = models[kwargs['model']]

    model = model_builder(input_shape=train_x.shape[1], **kwargs)
    model.compile(
        optimizer=opt,
        loss=rmse,
        metrics=['acc'],
    )
    model.summary()

    train(train_x, train_y, test_x, test_y, model, epochs=kwargs['epochs'])

    # score = model.test_on_batch(x=test_x, y=test_y)

    # print("Test score:", score)

    model.save(model_filename, overwrite=False)


if __name__ == "__main__":
    ap = ArgumentParser()

    hyp = ap.add_argument_group("hyper_parameters")
    hyp.add_argument('--lr', help="learning rate", type=int, default=0.05)
    hyp.add_argument('--epochs', help="Number of training epochs", type=int, default=10)

    hyp.add_argument('--batch_size', '-b', help="The batch size", type=int, default=100)
    hyp.add_argument('--sub_set', '-T', dest='sub_set', type=int,
                     help="The number of samples to draw from the training and test set",
                     required=False)

    ap.add_argument('--word-embedding', dest="word_embedding", type=bool, default=False)
    ap.add_argument('--model-filename', '-f', dest="model_filename", type=str, default="deep_model.h5")
    ap.add_argument('-c', '--config', dest='config', required=False)
    ap.add_argument('--model', default="simple")
    ap.add_argument('--kaggle', type=bool, default=False)

    args = ap.parse_args()
    kwargs = vars(args)

    if args.config:
        if not os.path.isfile(args.config):
            raise ValueError(args.config + " is not a file.")
        stream = open(args.config, 'r')
        config = yaml.load(stream, Loader=yaml.Loader)
        kwargs.update(config)

    sub_set = kwargs.get('sub_set', 30000)
    mdf = kwargs.get("mdf", 50)
    VALIDATION_SPLIT = kwargs.get("validation_split", 0.25)
    MAX_NB_WORDS = kwargs.get("max_nb_words", 1000)
    MAX_SEQUENCE_LENGTH = kwargs.get("max_sequence_length", 1000)
    EMBEDDING_DIM = kwargs.get("embedding_dim", 10000)
    GLOVE_DIR = kwargs.get("glove_dir", "180")
    figure_filename = kwargs.get("figure_filename", "fig.png")
    
    print("kwargs", kwargs)

    if args.kaggle:
        kaggle_test(kwargs['model_filename'])
    else:
        main(**kwargs)

