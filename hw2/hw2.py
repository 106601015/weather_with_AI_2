import os
import numpy as np
import datetime
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import save_model, load_model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, GlobalMaxPooling2D, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import cv2

img_row, img_col, num_class = 32, 32, 10

# load data, nomalize and to_categorical
def get_data_x_y(train_or_test='train'):
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
    print('---> data_x.shape, data_y.shape, type(data_x):', data_x.shape, data_y.shape, type(data_x))
    return data_x, data_y

def create_VGG16_model_complete():
    VGG16_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(img_row, img_col, 3)
    )

    VGG16_model.trainable = False

    model = Sequential()
    model.add(VGG16_model)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    #model.summary()
    return model

def create_VGG16_model_finetuning():
    VGG16_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(img_row, img_col, 3)
    )

    VGG16_model.trainable = True
    set_trainable = False
    for layer in VGG16_model.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    model = Sequential()
    model.add(VGG16_model)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.summary()
    return model

########################################################################

def create_VGG16_model_complete_64():
    VGG16_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(64, 64, 3)
    )

    VGG16_model.trainable = False

    model = Sequential()
    model.add(VGG16_model)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    #model.summary()
    return model

def create_VGG16_model_finetuning_64():
    VGG16_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(64, 64, 3)
    )

    VGG16_model.trainable = True
    set_trainable = False
    for layer in VGG16_model.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    model = Sequential()
    model.add(VGG16_model)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    #model.summary()
    return model

########################################################################

def create_cnn_model_v1():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_row, img_col, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(Dropout(0.1))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dropout(0.1))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(10, activation='softmax'))
    #model.summary()
    return model

def create_cnn_model_v2():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_row, img_col, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dropout(0.1))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(10, activation='softmax'))
    #model.summary()
    return model

def create_cnn_model_v3():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_row, img_col, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dropout(0.1))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(10, activation='softmax'))
    #model.summary()
    return model

def create_cnn_model_v4():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_row, img_col, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    #model.summary()
    return model

def create_cnn_model_v5():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(img_row, img_col, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    #model.summary()
    return model

def create_cnn_model_v6():
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=3, input_shape=(img_row, img_col, 3), activation='relu', padding='same'))
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(rate=0.25))
    model.add(Dense(10, activation='softmax'))
    #model.summary()
    return model

def create_cnn_model_v7():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(img_row, img_col, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(48, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(48, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(80, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(80, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(80, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(80, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(80, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))

    model.add(GlobalMaxPooling2D())
    model.add(Dropout(0.25))
    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))

    #model.summary()
    return model

def show_training_curve(train_history, model_name):
    plt.plot(train_history.history['loss'])
    plt.plot(train_history.history['val_loss'])
    plt.title('train history')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['loss', 'val_loss'], loc='upper left')
    plt.savefig(os.path.join('.', 'hw2', model_name))
    plt.close()


def main(operator):
    print('-----main {} init-----'.format(operator))

    hw2_path = os.path.join('.', 'hw2')
    model_dict = {
        'cnn_v1':create_cnn_model_v1(),
        'cnn_v2':create_cnn_model_v2(),
        'cnn_v3':create_cnn_model_v3(),
        'cnn_v4':create_cnn_model_v4(),
        'cnn_v5':create_cnn_model_v5(),
        'cnn_v6':create_cnn_model_v6(),
        'cnn_v7':create_cnn_model_v7(),
        'VGG16_pretrain':create_VGG16_model_complete(),
        'VGG16_pretrain_finetuning':create_VGG16_model_finetuning(),
    }
    model_dict_64 = {
        'VGG16_pretrain_64':create_VGG16_model_complete_64(),
        'VGG16_pretrain_finetuning_64':create_VGG16_model_finetuning_64(),
    }

    if operator == 'train':
        train_data_x, train_data_y = get_data_x_y(train_or_test='train')

        # 10 epochs
        for model_name, model in model_dict.items():
            if os.path.isfile(os.path.join(hw2_path, '{}_10epochs.h5'.format(model_name))):
                print('{}_10epochs has been train!!!'.format(model_name))
            else:
                print('--------->', model_name, '10 epochs gogo')
                model.compile(
                    loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy']
                )
                earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
                train_history = model.fit(
                    train_data_x, train_data_y,
                    batch_size=32,
                    epochs=10,
                    verbose=1,
                    validation_split=0.1,
                    callbacks=[earlystop],
                )
                save_model(model, os.path.join(hw2_path, '{}_10epochs.h5'.format(model_name)))
                show_training_curve(train_history, model_name+'_10epochs')

        # 20 epochs
        for model_name, model in model_dict.items():
            if os.path.isfile(os.path.join(hw2_path, '{}_20epochs.h5'.format(model_name))):
                print('{}_20epochs has been train!!!'.format(model_name))
            else:
                model.compile(
                    loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy']
                )
                earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
                train_history = model.fit(
                    train_data_x, train_data_y,
                    batch_size=32,
                    epochs=20,
                    verbose=1,
                    validation_split=0.1,
                    callbacks=[earlystop],
                )
                save_model(model, os.path.join(hw2_path, '{}_20epochs.h5'.format(model_name)))
                show_training_curve(train_history, model_name+'_20epochs')


        train_data_x = np.array([cv2.resize(i, (64, 64)) for i in train_data_x])
        print('-----> train_data_x.shape, train_data_y.shape:', train_data_x.shape, train_data_y.shape)
        # 10 epochs + 64
        for model_name, model in model_dict_64.items():
            if os.path.isfile(os.path.join(hw2_path, '{}_10epochs.h5'.format(model_name))):
                print('{}_10epochs has been train!!!'.format(model_name))
            else:
                print('--------->', model_name, '10 epochs gogo')
                model.compile(
                    loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy']
                )
                earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
                train_history = model.fit(
                    train_data_x, train_data_y,
                    batch_size=32,
                    epochs=10,
                    verbose=1,
                    validation_split=0.1,
                    callbacks=[earlystop],
                )
                save_model(model, os.path.join(hw2_path, '{}_10epochs.h5'.format(model_name)))
                show_training_curve(train_history, model_name+'_10epochs')

        # 20 epochs + 64
        for model_name, model in model_dict_64.items():
            if os.path.isfile(os.path.join(hw2_path, '{}_20epochs.h5'.format(model_name))):
                print('{}_20epochs has been train!!!'.format(model_name))
            else:
                model.compile(
                    loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy']
                )
                earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
                train_history = model.fit(
                    train_data_x, train_data_y,
                    batch_size=32,
                    epochs=20,
                    verbose=1,
                    validation_split=0.1,
                    callbacks=[earlystop],
                )
                save_model(model, os.path.join(hw2_path, '{}_20epochs.h5'.format(model_name)))
                show_training_curve(train_history, model_name+'_20epochs')



    elif operator == 'test':
        test_data_x, test_data_y = get_data_x_y(train_or_test='test')
        for file_name in os.listdir(hw2_path):
            if file_name.split('.')[-1] == 'h5' and file_name.split('_')[-2][-2:] != '64':
                model = load_model(os.path.join(hw2_path, file_name))
                score = model.evaluate(test_data_x, test_data_y, verbose=0)
                print('{} test acc: {:.5f}, test loss: {:.5f}'.format(file_name, score[1], score[0]))

        test_data_x = np.array([cv2.resize(i, (64, 64)) for i in test_data_x])
        for file_name in os.listdir(hw2_path):
            if file_name.split('.')[-1] == 'h5' and file_name.split('_')[-2][-2:] == '64':
                model = load_model(os.path.join(hw2_path, file_name))
                score = model.evaluate(test_data_x, test_data_y, verbose=0)
                print('{} test acc: {:.5f}, test loss: {:.5f}'.format(file_name, score[1], score[0]))
    else:
        print('operator error!!!')

if __name__ == '__main__':
    main(operator='train')
    main(operator='test')