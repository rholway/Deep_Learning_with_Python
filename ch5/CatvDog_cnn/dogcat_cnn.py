import cv2
import os
import glob

from keras import layers
from keras import models
from keras import optimizers

dog_dir = '/Users/ryanholway/Desktop/CatvDog_cnn/data/cat_dog/dog'
cat_dir = '/Users/ryanholway/Desktop/CatvDog_cnn/data/cat_dog/cat'

dog_path = os.path.join(dog_dir, '*g')
cat_path = os.path.join(cat_dir, '*g')

dog_files = glob.glob(dog_path)
cat_files = glob.glob(cat_path)

dog_pics = []
for d in dog_files:
    d_img = cv2.imread(d)
    dog_pics.append(d_img)

cat_pics = []
for c in cat_files:
    c_img = cv2.imread(c)
    cat_pics.append(c_img)

train_dogs_dir = dog_pics[:100]
train_cats_dir = cat_pics[:100]
val_dogs_dir = dog_pics[100:150]
val_cats_dir = cat_pics[100:150]
test_dogs_dir = dog_pics[150:]
test_cats_dir = cat_pics[150:]

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
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
                optimizer=optimizers.RMSprop(lr=1e-4),
                metrics=['acc'])
