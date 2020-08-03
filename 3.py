import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD

# Part where MNIST data set is loaded
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#################################################
# Adding noise
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

################################################

# Reshaping the data set
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Encoding target values
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Scaling pixel values for modeling
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train / 255.0
x_test = x_test / 255.0

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

# Create the model
denoiser = Sequential()
denoiser.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
denoiser.add(Conv2DTranspose(64, kernel_size=(3,3), activation='relu', kernel_initializer='he_uniform'))
denoiser.add(Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same'))

# Compile and fit data
denoiser.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
denoiser.fit(x_train_noisy, x_train,
                epochs=5,
                batch_size=64,
                validation_split=0.2)

x_denoised_train = denoiser.predict(x_train_noisy)
x_denoised_test = denoiser.predict(x_test_noisy)

# Evaluate the model
history = model.fit(x_denoised_train, y_train, epochs=10, batch_size=32, validation_data=(x_denoised_test, y_test))

test_loss, test_acc = model.evaluate(x_denoised_test, y_test)
print('Accuracy:', test_acc)