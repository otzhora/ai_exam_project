import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.datasets import mnist
from keras import backend as K


def train_model(model_name):
    json_file = open(model_name+"_arch.json", 'r')
    loaded_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_json)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    if model_name == 'basic' or model_name == 'basic3':
        epoch = 20
        if model_name == 'basic3':
            epoch = 100

        x_train = x_train.reshape(60000, 784).astype('float32') / 255
        x_test = x_test.reshape(10000, 784).astype('float32') / 255

        model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(), metrics=['accuracy'])

        model.fit(x_train, y_train, batch_size=128,
                  epochs=epoch, verbose=1, validation_split=0.2)

        score = model.evaluate(x_test, y_test, verbose=1)
        print("Test score: ", score[0])  # basic: 0.3484541090369225  basic3: 0.09169222676060163
        print("Test acc: ", score[1])  # basic: 0.9062  basic3: 0.9711

        model.save_weights('weigths/'+model_name+'_weig.h5')

    if model_name == 'LeNet':
        K.set_image_dim_ordering('th')
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        x_train = x_train[:, np.newaxis, :, :]
        x_test = x_test[:, np.newaxis, :, :]

        model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(), metrics=['accuracy'])
        model.fit(x_train, y_train, batch_size=128,
                  epochs=6, verbose=1, validation_split=0.2)

        score = model.evaluate(x_test, y_test, verbose=1)
        print("Test score: ", score[0])  # 0.09544988954290748
        print("Test acc: ", score[1])  # 0.9723

        model.save_weights('weigths/' + model_name + '_weig.h5')


if __name__ == '__main__':
    model_name = 'LeNet'
    train_model(model_name)
