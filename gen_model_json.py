import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K


def gen_model(name):
    model = 0
    if name == 'basic':
        classes = 10
        input_shape = 784
        model = Sequential()

        model.add(Dense(classes, input_shape=(input_shape, )))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(), metrics=['accuracy'])

    if name == 'basic3':
        classes = 10
        input_shape = 784
        hidden = 128
        model = Sequential()

        model.add(Dense(hidden, input_shape=(input_shape,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(hidden))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(), metrics=['accuracy'])

    if model_name == 'LeNet':
        classes = 10
        input_shape = (1, 28, 28)

        model = Sequential()
        model.add(Conv2D(20, kernel_size=5, padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(50, kernel_size=5, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))

        model.add(Dense(classes))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(), metrics=['accuracy'])
    return model


if __name__ == '__main__':
    model_name = 'LeNet'
    if model_name == 'LeNet':
        K.set_image_dim_ordering('th')
    model = gen_model(model_name)
    model.summary()
    open(model_name+"_arch.json", 'w').write(model.to_json())
