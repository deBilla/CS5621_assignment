import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD

# Part where MNIST data set is loaded
(x_train, y_train), (x_test, y_test) = mnist.load_data()

noise_factor = 0.25

x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Reshaping the data set
x_train_noisy = x_train_noisy.reshape((x_train_noisy.shape[0], 28, 28, 1))
x_test_noisy = x_test_noisy.reshape((x_test.shape[0], 28, 28, 1))

# Scaling pixel values for modeling
x_train_noisy = x_train_noisy.astype('float32')
x_test_noisy = x_test_noisy.astype('float32')

x_train_noisy = x_train_noisy / 255.0
x_test_noisy = x_test_noisy / 255.0

# Encoding target values
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile model
opt = SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Evaluate the model
history = model.fit(x_train_noisy, y_train, epochs=10, batch_size=32, validation_data=(x_test_noisy, y_test))

test_loss, test_acc = model.evaluate(x_test_noisy, y_test)
print('Accuracy:', test_acc)