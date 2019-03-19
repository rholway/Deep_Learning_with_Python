import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np



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
    f = open(os.path.join(glove_dir, 'globe.6B.100d.txt'))
    for line in f:
        values = line.split()
        words = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print(f'Found {len(embeddings_index)} word vectors.')
