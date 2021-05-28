import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Flatten, Dense, GRU, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping

lookback = 720  # forecast with five days of data (1440)
step = 6       # our observations will be sampled at one data point per hour.
delay = 144     # our targets will be 24 hours in the future.
batch_size = 128
'''
data: The original array of floating point data, which we just normalized in the code snippet above.
lookback: How many timesteps back should our input data go.
delay: How many timesteps in the future should our target be.
min_index and max_index: Indices in the data array that delimit which timesteps to draw from. This is useful for keeping a segment of the data for validation and another one for testing.
shuffle: Whether to shuffle our samples or draw them in chronological order.
batch_size: The number of samples per batch.
step: The period, in timesteps, at which we sample data. We will set it 6 in order to draw one data point every hour.
'''
# data generator
def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback

    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets
def reverse_order_generator(data, lookback, delay, min_index, max_index,
                            shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples[:, ::-1, :], targets

# baseline evaluate
def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print('---> evaluate_naive_method:', np.mean(batch_maes))

def show_training_curve(train_history, model_name):
    plt.plot(train_history.history['loss'])
    plt.plot(train_history.history['val_loss'])
    plt.title('train history')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['loss', 'val_loss'], loc='upper left')
    plt.savefig(os.path.join('.', 'RNNtest', model_name))
    plt.close()


def create_simple_DNN():
    model = Sequential()
    model.add(Flatten(input_shape=(lookback // step, float_data.shape[-1])))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    return model
def create_simgle_GRU():
    model = Sequential()
    model.add(GRU(32, input_shape=(None, float_data.shape[-1])))
    model.add(Dense(1))
    return model
def create_dropout_GRU():
    model = Sequential()
    model.add(GRU(
        32,
        dropout=0.2,
        recurrent_dropout=0.2,
        input_shape=(None, float_data.shape[-1])))
    model.add(Dense(1))
    return model
def create_stacking_GRU():
    model = Sequential()
    model.add(GRU(
        32,
        dropout=0.1,
        recurrent_dropout=0.5,
        return_sequences=True,
        input_shape=(None, float_data.shape[-1])))
    model.add(GRU(
        64,
        activation='relu',
        dropout=0.1,
        recurrent_dropout=0.5))
    model.add(Dense(1))
    return model
def create_reverse_GRU():
    model = Sequential()
    model.add(GRU(32, input_shape=(None, float_data.shape[-1])))
    model.add(Dense(1))
    return model
def create_1dcnn_v1():
    model = Sequential()
    model.add(Conv1D(32, 7, activation='relu', input_shape=(None, float_data.shape[-1])))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(32, 7, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1))
    return model
def create_1dcnn_v2():
    model = Sequential()
    model.add(Conv1D(32, 5, activation='relu', input_shape=(None, float_data.shape[-1])))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(32, 5, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(32, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1))
    return model
def create_1dcnn_GRU():
    model = Sequential()
    model.add(Conv1D(32, 5, activation='relu',
                        input_shape=(None, float_data.shape[-1])))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(32, 5, activation='relu'))
    model.add(GRU(32, dropout=0.1, recurrent_dropout=0.5))
    model.add(Dense(1))
    return model

def main(operator):
    temp_pre_test_path = os.path.join('.', 'temp_pre_test')
    model_dict = {
        'simple_DNN':create_simple_DNN(),
        'simgle_GRU':create_simgle_GRU(),
        'dropout_GRU':create_dropout_GRU(),
        'stacking_GRU':create_stacking_GRU(),
        'reverse_GRU':create_reverse_GRU(),
        '1dcnn_v1':create_1dcnn_v1(),
        '1dcnn_v2':create_1dcnn_v2(),
        '1dcnn_GRU':create_1dcnn_GRU(),
    }
    if operator == 'train':
        for model_name, model in model_dict.items():
            if os.path.isfile(os.path.join(temp_pre_test_path, '{}_10epochs.h5'.format(model_name))):
                print('{}_10epochs has been train!!!'.format(model_name))
            else:
                print('--------->', model_name, '10 epochs gogo')
                model.compile(
                    loss='mae',
                    optimizer='adam', #'rmsprop'
                    #metrics=['accuracy'] #['acc']
                )
                earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

                if model_name=='reverse_GRU':
                    train_history = model.fit_generator(
                        train_gen_reverse,
                        steps_per_epoch=500,
                        epochs=10,
                        validation_data=val_gen,
                        validation_steps=val_steps
                    )
                else:
                    train_history = model.fit_generator(
                        train_gen,
                        steps_per_epoch=500,
                        epochs=10,
                        validation_data=val_gen,
                        validation_steps=val_steps
                    )

                save_model(model, os.path.join(temp_pre_test_path, '{}_10epochs.h5'.format(model_name)))
                show_training_curve(train_history, model_name+'_10epochs')

    elif operator == 'test':
        for file_name in os.listdir(temp_pre_test_path):
            if file_name.split('.')[-1] == 'h5':
                model = load_model(os.path.join(temp_pre_test_path, file_name))
                score = model.evaluate_generator(test_gen, test_steps) ###############
                print('{} maes score:', score)
                #print('{} test acc: {:.5f}, test loss: {:.5f}'.format(file_name, score[1], score[0]))
    else:
        print('operator error!!!')


if __name__ == '__main__':
    f = open(os.path.join('.', 'temp_pre_test', 'jena_climate_2009_2016.csv'))
    data = f.read()
    f.close()

    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:]
    print('---> header:', header)
    print('---> lines[0]:', lines[0])

    float_data = np.zeros((len(lines), len(header)-1))
    for i, line in enumerate(lines):
        values = [float(x) for x in line.split(',')[1:]]
        float_data[i, :] = values
    print('---> pre float_data:', float_data.shape, float_data[0])

    mean = float_data[:200000].mean(axis=0)
    float_data -= mean
    std = float_data[:200000].std(axis=0)
    float_data /= std
    print('---> aft float_data:', float_data.shape, float_data[0])

    train_gen = generator(
            float_data,
            lookback=lookback,
            delay=delay,
            min_index=0,
            max_index=200000,
            shuffle=True,
            step=step,
            batch_size=batch_size
        )
    val_gen = generator(
            float_data,
            lookback=lookback,
            delay=delay,
            min_index=200001,
            max_index=300000,
            step=step,
            batch_size=batch_size
        )
    test_gen = generator(
            float_data,
            lookback=lookback,
            delay=delay,
            min_index=300001,
            max_index=None,
            step=step,
            batch_size=batch_size
        )

    train_gen_reverse = reverse_order_generator(
            float_data,
            lookback=lookback,
            delay=delay,
            min_index=0,
            max_index=200000,
            shuffle=True,
            step=step,
            batch_size=batch_size
        )
    val_gen_reverse = reverse_order_generator(
            float_data,
            lookback=lookback,
            delay=delay,
            min_index=200001,
            max_index=300000,
            step=step,
            batch_size=batch_size
        )

    # This is how many steps to draw from `val_gen`
    # in order to see the whole validation set:
    val_steps = (300000 - 200001 - lookback) // batch_size
    # This is how many steps to draw from `test_gen`
    # in order to see the whole test set:
    test_steps = (len(float_data) - 300001 - lookback) // batch_size

    main('train')
    main('test')
    evaluate_naive_method()