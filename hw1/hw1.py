from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import save_model, load_model, Sequential
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping
#from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
import datetime
import os
import numpy as np

#TODO: RGB->single or create 3turnle

# load data, nomalize and to_categorical
def get_data_x_y(train_or_test='train'):
    img_row, img_col, num_class = 32, 32, 10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    if train_or_test == 'train':
        data_x = x_train
        data_y = y_train
    elif train_or_test == 'test':
        data_x = x_test
        data_y = y_test
    else:
        print('get_data_x_y train_or_test error!!!')
    data_x = data_x.astype('float32') / 255.0   # nomalize
    data_y = to_categorical(data_y, num_class)  # to_categorical
    #print('data_x.shape, data_y.shape, type(data_x):', data_x.shape, data_y.shape, type(data_x))
    return data_x, data_y

def create_dnn_model_v1():
    model = Sequential()

    model.add(tf.keras.Input(shape=(32, 32, 3), batch_size=None))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    return model

'''
# Flatten end
def create_dnn_model_v2():
    model = Sequential()

    model.add(tf.keras.Input(shape=(32, 32, 3), batch_size=None))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.summary()
    return model
'''

# add Dropout
def create_dnn_model_v3():
    model = Sequential()

    model.add(tf.keras.Input(shape=(32, 32, 3), batch_size=None))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    return model

# easyrize
def create_dnn_model_v4():
    model = Sequential()

    model.add(tf.keras.Input(shape=(32, 32, 3), batch_size=None))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    return model

# parallel
def create_dnn_model_v5():
    model = Sequential()

    model.add(tf.keras.Input(shape=(32, 32, 3), batch_size=None))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    return model

# sample
def create_dnn_model_v6():
    model = Sequential()

    model.add(tf.keras.Input(shape=(32, 32, 3), batch_size=None))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    #model.add(Dense(128, activation = 'relu', input_shape = (32*32*3, )))

    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(1024, activation = 'relu'))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))
    model.summary()
    return model

def create_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dropout(0.1))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(10, activation='softmax'))
    model.summary()
    return model

# plot diagnostic learning curves
def summarize_diagnostics(history, model_name):
    # plot loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    plt.legend()
    # plot accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    plt.legend()
    # save plot to file
    plt.savefig('.\\hw1\\{}_{}'.format(model_name, datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")))
    plt.close()


if __name__ == '__main__':
    print('__main__ gogo')
    x_train, y_train = get_data_x_y(train_or_test='train')
    x_test, y_test = get_data_x_y(train_or_test='test')
    dnn_model_v1 = create_dnn_model_v1() #best acc: 49.390
    #dnn_model_v2 = create_dnn_model_v2() #too big to train!!
    dnn_model_v3 = create_dnn_model_v3() #best acc: 45.100
    dnn_model_v4 = create_dnn_model_v4() #best acc: 50.090
    dnn_model_v5 = create_dnn_model_v5() #best acc: 10.000, fail try
    dnn_model_v6 = create_dnn_model_v6() #best acc: ????
    #cnn_model = create_cnn_model() #acc: 69.900
    model_dict = {
        #'dnn_v1':dnn_model_v1,
        ##'dnn_v2':dnn_model_v2,
        #'dnn_v3':dnn_model_v3,
        #'dnn_v4':dnn_model_v4,
        #'dnn_v5':dnn_model_v5,
        'dnn_v6':dnn_model_v6,
        #'cnn':cnn_model,
    }
    for model_name, model in model_dict.items():
        # train
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #train_history = model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_test, y_test), verbose=1)
        earlystop = EarlyStopping(monitor="val_loss", patience = 10, verbose = 1)
        train_history = model.fit(x_train, y_train, epochs=20, batch_size=64, callbacks=[earlystop], shuffle=True, validation_split=0.2, verbose=1)
        #callbacks=[earlystop], shuffle=True, validation_split=0.2 (no validation_data)
        # test
        _, acc = model.evaluate(x_test, y_test, verbose=0)
        print('-----> acc: %.3f <-----' % (acc * 100.0))
        summarize_diagnostics(train_history, model_name)
        # save
        save_model(model, '.\\hw1\\{}_{}.h5'.format(model_name, datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")))