import numpy as np
import tensorflow as tf
import os # this is to walk through the folder
import random

from tqdm import tqdm
# tqdm shows the progress bar during the loop execution

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

# controlling random seed.
seed = 42
np.random.seed = seed

TRAIN_PATH = 'C:/Users/Nacho/Documents/Coding/DIC/Project/data/train/'
TEST_PATH = 'C:/Users/Nacho/Documents/Coding/DIC/Project/data/test/'

train_ids = next(os.walk(TRAIN_PATH + "images"))[2]
test_ids = next(os.walk(TEST_PATH + "images"))[2]
# next: returns the next item from the iterator.
# walk: spits out every name of files.
# tuple [2]: this is return the files name.

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8) # empty array
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool) # empty array
# As reading and resizing each images, it updates empty array with a new number.

print('Resizing training images and masks')
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)): # enumerate 'train_ids'
    path = (TRAIN_PATH + '/images/' + id_) # path to the image
    img = imread(path)[:,:,:IMG_CHANNELS] # read the images inside 'images' dir
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True) # resize the images
    X_train[n] = img # Fill empty X_train with values from images.

# print('Resizing X_train is done')

# mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)[:,:,1] # empty array for masks
    mask = img_to_array(load_img(TRAIN_PATH + '/masks/' + id_))[:,:,1]
    Y_train[n] = resize(mask, (128, 128, 1), mode='constant', preserve_range=True)

# print('Done!')

print("X_train: {}".format(X_train.shape))
print("Y_train: {}".format(Y_train.shape))

# test images
# test dir has no masks so this is only to load images, resize them, and put them into the array.
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8) # empty array for X_test
sizes_test = []
print('Resizing test images')
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = (TEST_PATH + '/images/' + id_)
    img = imread(path)[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

# print('Done!')
print('X_test: {}'.format(X_test.shape))

########## Sanity Check ##########
'''
image_x = random.randint(0, len(train_ids)) # pick random number between 0 to 670
imshow(X_train[image_x]) # use the above number as an index to load an image.
plt.show()
imshow(np.squeeze(Y_train[image_x]))
plt.show()
'''

########## Build the model ##########
inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))

# Converting 8-bit integers to float by dividing 255 using python lambda function.
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

print("s.shape: {}".format(s.shape))

########## Contracting or Encoding Path ##########
# Activation, using relu is for
# Initializer is the starting value of the weight. normal distribution is the Gausian distribution; centered around zero.
# Padding is the same because we want the input and output, the same resolution.

# First set of convolutional layers
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1) # 10% Dropout between conv step to avoid over filling
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

# Second set of convolutional layers
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

# Third set of convolutional layers
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

# Fourth set of convolutional layers
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

# Fifth and bottom set of convolutional layers
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
# No MaxPooling2D at a bottom of U-net

########## Expansive or the Decoding Path ##########
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1])
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = tf.keras.layers.Conv2D (1, (1, 1), activation='sigmoid') (c9)

model = tf.keras.Model (inputs=[inputs], outputs=[outputs])

# Optimization is a lot of back propagation algorithms to train the model.
# Adam is the popular one. The traditional one is such like stochastic gradient descent a.k.a. SGD.
# This is a binary classification so the loss function will be the binary_crossentropy.
# Optimizer is trying to minimize the loss function until it reaches the minimum.
model.compile (optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary ()

# Checkpoint is essential for saving your process in case of error.
# Too many epochs = over fitting.
# Callback is essential for the EarlyStopping to monitor 'val_loss' with patience about 3
# TensorBoard is useful to graphically understand the progress.

########## Model Checkpoint ##########
# Save model every epochs and leave the best only.
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_forest.h5', verbose=1, save_best_only=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss'),
    # Monitor the validation loss parameter and if the value doesn't get better after 3 epochs further, stop.
    tf.keras.callbacks.TensorBoard(log_dir='logs')]
    # To graphically look at various thing using TensorBoard.

results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=100, callbacks=callbacks)
# Fit pixel values from X and y to the model.
# Holding out 10% of data for the validation.

########## Once its done... ##########
########## Sanity Check on prediction ##########
idx = random.randint(0, len(train_ids)) # pick random number between 0 to 670

# Predict the random image selected from above.
# Below will return value in between 0 to 1.
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
# Predicting X_train[603] -> (603, 128, 128, 3)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
# Predicting X_train[603] -> (603, 128, 128, 3)
preds_test = model.predict(X_test, verbose=1)

# applying threshold to make everything binary for all train, val, and test pixels.
preds_train_t = (preds_train > 0.5).astype(np.uint8) # anything that is over 0.5, convert that into 1.
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

########## Perform a sanity check on some random training samples ##########
# randomly selecting an image to visualize training samples.
ix = random.randint(0, len(preds_train_t))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()

########## Perform a sanity check on some random validation samples ##########
# randomly selecting an image to visualize validation samples.
ix = random.randint(0, len(preds_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()

