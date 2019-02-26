from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt

def vect_seq(sequences, dimension=5000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000)
    x_train = vect_seq(x_train)
    x_test = vect_seq(x_test)
    y_train = np.asarray(y_train).astype('float32')
    y_test = np.asarray(y_test).astype('float32')

    # translate indexes to words
    # word_index = imdb.get_word_index()
    # reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])
    # decoded_review = ' '.join([reverse_word_index.get(i-3, '?') for i in x_train[0]])

    # make network of layers
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(5000,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))


    # validation set
    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]
    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]

    # train/fit model
    model.compile(optimizer='rmsprop',
                    loss='binary_crossentropy',
                    metrics=['acc'])
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=20,
                        batch_size=512,
                        validation_data=(x_val, y_val)
                        )

    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    '''
    # plot loss
    epochs = range(1, len(history_dict['acc']) + 1)
    plt.plot(epochs, loss_values, 'bo', label= 'Training loss')
    plt.plot(epochs, val_loss_values, 'b', label= 'Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('bi_clf_train_and_val_loss')
    plt.close()

    # plot accuracy
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']
    plt.plot(epochs, acc_values, 'bo', label='Training acc')
    plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('bi_clf_train_and_val_acc')
    plt.close()
    '''
