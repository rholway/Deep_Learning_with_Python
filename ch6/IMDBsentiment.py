import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # process the labels of the raw IMDB data
    imdb_dir = '/Users/ryanholway/Desktop/data_science/deep_learning_with_python/data/IMDB/aclImdb'
    train_dir = os.path.join(imdb_dir, 'train')

    labels = []
    texts = []

    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(train_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name, fname))
                texts.append(f.read())
                f.close()
                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)

    # tokenizing the text of the raw IMDB data
    maxlen = 100
    training_samples = 200
    validation_samples = 10000
    max_words = 10000

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print(f'Found {len(word_index)} unique tokens.')

    data = pad_sequences(sequences, maxlen=maxlen)

    labels = np.asarray(labels)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    x_train = data[:training_samples]
    y_train = labels[:training_samples]
    x_val = data[training_samples: training_samples + validation_samples]
    y_val = labels[training_samples: training_samples + validation_samples]

    # parse the GloVe word embeddings file
    glove_dir = '/Users/ryanholway/Desktop/data_science/deep_learning_with_python/data/GloVe/glove.6B'

    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print(f'Found {len(embeddings_index)} word vectors.')

    # prepare GloVe word-embedding matrix
    embedding_dim = 100

    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    # define the model
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    # load the GloVe matrix into the Embedding layer - the first layer
    # we want to keep the same weights from the pretrained model, not update
    # or train the weights, so we set the trainable attribute to false
    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False

    # compile, train, and evaluate
    model.compile(optimizer='rmsprop',
                    loss='binary_crossentropy',
                    metrics=['acc'])
    history = model.fit(x_train, y_train,
                        epochs=10,
                        batch_size=32,
                        validation_data=(x_val, y_val))
    model.save_weights('pre_trained_glove_model.h5')

    # plot the model performance
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig('train_val_acc')

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('train_val_loss')

    plt.show()
