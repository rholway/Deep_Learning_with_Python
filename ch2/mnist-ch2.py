from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((60000, 28*28)).astype('float32') / 255
    x_test = x_test.reshape((10000, 28*28)).astype('float32') / 255
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5, batch_size=128)

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('test_loss', test_loss, 'test_acc', test_acc)
