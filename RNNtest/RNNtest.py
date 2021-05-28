import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, LSTM
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping

max_features = 10000  # number of words to consider as features
maxlen = 500  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

def get_data_x_y(train_or_test='train'):
    (train_data_x, train_data_y), (test_data_x, test_data_y) = imdb.load_data(num_words=max_features)
    train_data_x = sequence.pad_sequences(train_data_x, maxlen=maxlen)
    test_data_x = sequence.pad_sequences(test_data_x, maxlen=maxlen)
    print('----> train_data_x/test_data_x shape:', train_data_x.shape, test_data_x.shape)

    if train_or_test=='train':
        return train_data_x, train_data_y
    elif train_or_test=='test':
        return test_data_x, test_data_y
    else:
        print('get_data_x_y wtf??')

def create_simple_RNN():
    model = Sequential()
    '''
    model.add(Embedding(10000, 32))
    model.add(SimpleRNN(32, return_sequences=True))
    model.add(SimpleRNN(32, return_sequences=True))
    model.add(SimpleRNN(32, return_sequences=True))
    model.add(SimpleRNN(32))
    '''
    model.add(Embedding(max_features, 32))
    model.add(SimpleRNN(32))
    model.add(Dense(1, activation='sigmoid'))
    #model.summary()
    return model

def create_LSTM():
    model = Sequential()
    model.add(Embedding(max_features, 32))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))
    #model.summary()
    return model

def show_training_curve(train_history, model_name):
    plt.plot(train_history.history['loss'])
    plt.plot(train_history.history['val_loss'])
    plt.title('train history')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['loss', 'val_loss'], loc='upper left')
    plt.savefig(os.path.join('.', 'RNNtest', model_name))
    plt.close()

def main(operator):
    RNNtest_path = os.path.join('.', 'RNNtest')
    model_dict = {
        'simple_RNN':create_simple_RNN(),
        'LSTM':create_LSTM(),
    }
    if operator == 'train':
        train_data_x, train_data_y = get_data_x_y('train')
        for model_name, model in model_dict.items():
            if os.path.isfile(os.path.join(RNNtest_path, '{}_10epochs.h5'.format(model_name))):
                print('{}_10epochs has been train!!!'.format(model_name))
            else:
                print('--------->', model_name, '10 epochs gogo')
                model.compile(
                    loss='binary_crossentropy',
                    optimizer='adam', #'rmsprop'
                    metrics=['accuracy'] #['acc']
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
                save_model(model, os.path.join(RNNtest_path, '{}_10epochs.h5'.format(model_name)))
                show_training_curve(train_history, model_name+'_10epochs')

    elif operator == 'test':
        test_data_x, test_data_y = get_data_x_y('test')
        for file_name in os.listdir(RNNtest_path):
            if file_name.split('.')[-1] == 'h5':
                model = load_model(os.path.join(RNNtest_path, file_name))
                score = model.evaluate(test_data_x, test_data_y, verbose=0)
                print('{} test acc: {:.5f}, test loss: {:.5f}'.format(file_name, score[1], score[0]))
    else:
        print('operator error!!!')



if __name__ == '__main__':
    main(operator='train')
    main(operator='test')