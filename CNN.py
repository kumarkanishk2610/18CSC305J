import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import os

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Show an example image
i = 30009
plt.imshow(X_train[i])
plt.show()
print(y_train[i])

# Grid size for plotting images
W_grid = 4
L_grid = 4

# Plot multiple images
fig, axes = plt.subplots(L_grid, W_grid, figsize=(25, 25))
axes = axes.ravel()
n_training = len(X_train)

for i in np.arange(0, L_grid * W_grid):
    index = np.random.randint(0, n_training) # pick a random number
    axes[i].imshow(X_train[index])
    axes[i].set_title(y_train[index])
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.4)

# Preprocessing
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
number_cat = 10
y_train = to_categorical(y_train, number_cat)
y_test = to_categorical(y_test, number_cat)

# Define model
cnn_model = Sequential([
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=X_train.shape[1:]),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),
    Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),
    Flatten(),
    Dense(units=1024, activation='relu'),
    Dense(units=1024, activation='relu'),
    Dense(units=10, activation='softmax')
])

# Compile model
cnn_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Fit the model
history = cnn_model.fit(X_train, y_train, batch_size=32, epochs=1, shuffle=True)

# Evaluate the model
evaluation = cnn_model.evaluate(X_test, y_test)
print('Test Accuracy: {}'.format(evaluation[1]))

# Make predictions
predicted_classes = cnn_model.predict(X_test)
classes_x = np.argmax(predicted_classes, axis=1)
y_test = y_test.argmax(1)

# Plot some predictions
L = 7
W = 7
fig, axes = plt.subplots(L, W, figsize=(12, 12))
axes = axes.ravel()
for i in np.arange(0, L * W):
    axes[i].imshow(X_test[i])
    axes[i].set_title('Prediction = {}\n True = {}'.format(classes_x[i], y_test[i]))
    axes[i].axis('off')
plt.subplots_adjust(wspace=1)

# Confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, classes_x)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True)

# Save the model
directory = os.path.join(os.getcwd(), 'saved_models')
if not os.path.isdir(directory):
    os.makedirs(directory)
model_path = os.path.join(directory, 'keras_cifar10_trained_model.h5')
cnn_model.save(model_path)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=90,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True
)

datagen.fit(X_train)

# Fit the model with data augmentation
cnn_model.fit_generator(datagen.flow(X_train, y_train, batch_size=32), epochs=2)

# Evaluate the augmented model
score = cnn_model.evaluate(X_test, y_test)
print('Test accuracy with data augmentation:', score[1])

# Save the model
directory = os.path.join(os.getcwd(), 'saved_models')
if not os.path.isdir(directory):
    os.makedirs(directory)
model_path = os.path.join(directory, 'keras_cifar10_trained_model_Augmentation.h5')
cnn_model.save(model_path)