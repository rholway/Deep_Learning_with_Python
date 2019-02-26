from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

def vectorize_sequences(sequences, dimension=10000):
    '''
    vectorize the data: cast the label list as in integer tensor
    '''
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

def to_one_hot(labels, dimension=46):
    '''
    categorical encoding - one hot encoding
    but, we can also just user the keras built-in way
    '''
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.0
    return results



if __name__ == '__main__':
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
    print(f'length of train data: {len(train_data)}')
    print(f'length of test data: {len(test_data)}')

    # to get newswires back to text
    word_index = reuters.get_word_index()
    reverse_word_index = dict([(value, key) for (key,value) in word_index.items()])
    decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
    print(f'train data text for one example {decoded_newswire}')

    # an example of train label, which will be between 1-46
    print(f'example of one train label {train_labels[10]}')

    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    # these are the y's
    one_hot_train_labels = to_categorical(train_labels)
    one_hot_test_labels = to_categorical(test_labels)
    # we can also define the y's like this, but if we do, we need to change the
    # loss function to 'sparse_categorical_crossentropy'

    # y_train = np.array(train_labels)
    # y_test = np.array(test_labels)

    # set aside validation set
    x_val = x_train[:1000]
    partial_x_train = x_train[1000:]
    y_val = one_hot_train_labels[:1000]
    partial_y_train = one_hot_train_labels[1000:]
    '''
    # make the model
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(46, activation='softmax'))

    model.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    # train the model for 20 epochs
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=20,
                        batch_size=512,
                        validation_data=(x_val, y_val))
    # results
    results = model.evaluate(x_test, one_hot_test_labels)
    print(f'results of first model {results}')

    # plot training and validation loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label ='Vvalidation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('train_and_val_loss')
    # plt.show()
    plt.close()

    # plot training and validation accuracy
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('train_and_val_acc')
    plt.close()
    # plt.show()
    '''
    # model begins to start overfitting after 9 epochs, so we can retrain a model
    # with only 9 epochs
    # make the new model2
    model2 = models.Sequential()
    model2.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
    model2.add(layers.Dense(64, activation='relu'))
    model2.add(layers.Dense(46, activation='softmax'))

    model2.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    # train the model for 9 epochs
    model2.fit(partial_x_train,
                        partial_y_train,
                        epochs=9,
                        batch_size=512,
                        validation_data=(x_val, y_val))
    # results from second model
    results2 = model2.evaluate(x_test, one_hot_test_labels)
    print(f'results of first model {results2}')

    # generate predictions on new data (with the new model)
    predictions = model2.predict(x_test)
    # each entry in predictions i a vector of length 46, with each being a
    # probability of being in a class
    print(f'each entry in predictions is a vecotr of length 46 {predictions[0].shape}')
    # the largest entry i sthe predicted class
    print(f'the prediction for the first test data point is class {predictions[0]}')
