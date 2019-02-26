import os, shutil
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Moving pictures from uncompressed file into training, validation, and test
    # directories
    # original_dataset_dir = '/Users/ryanholway/Desktop/data_science/deep_learning_with_python/data/catdogCNN/dogs-vs-cats/train'
    #
    # base_dir = '/Users/ryanholway/Desktop/data_science/deep_learning_with_python/data/catdogCNN/model_data'
    # os.mkdir(base_dir)
    #
    # train_dir = os.path.join(base_dir, 'train')
    # os.mkdir(train_dir)
    # validation_dir = os.path.join(base_dir, 'validation')
    # os.mkdir(validation_dir)
    # test_dir = os.path.join(base_dir, 'test')
    # os.mkdir(test_dir)
    #
    # train_cats_dir = os.path.join(train_dir, 'cats')
    # os.mkdir(train_cats_dir)
    #
    # train_dogs_dir = os.path.join(train_dir, 'dogs')
    # os.mkdir(train_dogs_dir)
    #
    # validation_cats_dir = os.path.join(validation_dir, 'cats')
    # os.mkdir(validation_cats_dir)
    #
    # validation_dogs_dir = os.path.join(validation_dir, 'dogs')
    # os.mkdir(validation_dogs_dir)
    #
    # test_cats_dir = os.path.join(test_dir, 'cats')
    # os.mkdir(test_cats_dir)
    #
    # test_dogs_dir = os.path.join(test_dir, 'dogs')
    # os.mkdir(test_dogs_dir)
    #
    # fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
    # for fname in fnames:
    #     src = os.path.join(original_dataset_dir, fname)
    #     dst = os.path.join(train_cats_dir, fname)
    #     shutil.copyfile(src, dst)
    #
    # fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
    # for fname in fnames:
    #     src = os.path.join(original_dataset_dir, fname)
    #     dst = os.path.join(validation_cats_dir, fname)
    #     shutil.copyfile(src, dst)
    #
    # fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
    # for fname in fnames:
    #     src = os.path.join(original_dataset_dir, fname)
    #     dst = os.path.join(test_cats_dir, fname)
    #     shutil.copyfile(src, dst)
    #
    # fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
    # for fname in fnames:
    #     src = os.path.join(original_dataset_dir, fname)
    #     dst = os.path.join(train_dogs_dir, fname)
    #     shutil.copyfile(src, dst)
    #
    # fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    # for fname in fnames:
    #     src = os.path.join(original_dataset_dir, fname)
    #     dst = os.path.join(validation_dogs_dir, fname)
    #     shutil.copyfile(src, dst)
    #
    # fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
    # for fname in fnames:
    #     src = os.path.join(original_dataset_dir, fname)
    #     dst = os.path.join(test_dogs_dir, fname)
    #     shutil.copyfile(src, dst)
    # =========================================================
    # Start creating the CNN
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu',
                            input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    # add dropout layer to prevent overfitting
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                    optimizer=optimizers.RMSprop(lr=1e-4),
                    metrics=['acc'])

    # read images from directories
    train_datagen = ImageDataGenerator(
                        rescale = 1./255,
                        rotation_range=40,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True)

    # don't augment the validation data!
    test_datagen = ImageDataGenerator(rescale = 1./255)

    train_generator = train_datagen.flow_from_directory(
            '/Users/ryanholway/Desktop/data_science/deep_learning_with_python/data/catdogCNN/model_data/train',
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
            '/Users/ryanholway/Desktop/data_science/deep_learning_with_python/data/catdogCNN/model_data/validation',
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary')

    # check output of the train generator
    # for data_batch, labels_batch in train_generator:
    #     print(data_batch.shape)
    #     print(labels_batch.shape)
    #     break

    history = model.fit_generator(
            train_generator,
            steps_per_epoch=100,
            epochs=50,
            validation_data=validation_generator,
            validation_steps=50)

    model.save('cats_and_dogs_small_2.h5')
    # =========================================================
    # Display curves of loss and accuracy
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig('train_and_val_acc')

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('train_and_val_loss')
    # =========================================================
    # Data augmentation
    datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
    # view augmented images
    fnames = [os.path.join(train_dogs_dir, fname) for
        fname in os.listdir(train_dogs_dir)]
    img_path = fnames[4]
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size=1):
        plt.figure(i)
        imgplot = plt.imshow(image.array_to_img(batch[0]))
        i += 1
        if i % 4 == 0:
            break
    plt.savefig('dog_augmentaiton_img')
