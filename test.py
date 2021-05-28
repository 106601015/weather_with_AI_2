from tensorflow.keras import layers, Input, applications, callbacks
from tensorflow.keras.models import Model

if __name__ == '__main__':
    # functional way to build the model
    '''
    input_tensor = Input(shape=(64,))
    x = layers.Dense(32, activation='relu')(input_tensor)
    x = layers.Dense(32, activation='relu')(x)
    output_tensor = layers.Dense(10, activation='softmax')(x)
    model = Model(input_tensor, output_tensor)
    model.summary()
    '''

    # GoogLeNet Inception try
    '''
    branch_a = layers.Conv2D(128, 1, activation='relu', strides=2)(x)
    branch_b = layers.Conv2D(128, 1, activation='relu')(x)
    branch_b = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_b)
    branch_c = layers.AveragePooling2D(3, strides=2)(x)
    branch_c = layers.Conv2D(128, 3, activation='relu')(branch_c)

    branch_d = layers.Conv2D(128, 1, activation='relu')(x)
    branch_d = layers.Conv2D(128, 3, activation='relu')(branch_d)
    branch_d = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_d)

    output = layers.concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1)
    '''

    # multi output
    '''
    input_layer = Input(shape=(None, ), dtype='int32', name='input')
    embedding_layer = layers.Embedding(256, 50000)(input_layer)
    x = layers.Conv1D(128, 5, activation='relu')(embedding_layer)
    x = layers.MaxPooling1D(5)(x)
    x = layers.Conv1D(128, 5, activation='relu')(x)
    x = layers.Conv1D(128, 5, activation='relu')(x)
    x = layers.MaxPooling1D(5)(x)
    x = layers.Conv1D(128, 5, activation='relu')(x)
    x = layers.Conv1D(128, 5, activation='relu')(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)

    age_pre = layers.Dense(1, name='age')(x)
    income_pre = layers.Dense(10, activation='softmax', name='income')(x)
    gender_pre = layers.Dense(1, activation='sigmoid', name='gender')(x)

    model = Model(input_layer, [age_pre, income_pre, gender_pre])
    model.summary()

    model.compile(
        optimizer='rmsprop',
        #loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'],
        #loss_weights=[0.25, 1., 10.],
        loss={
            'age_pre':'mse',
            'income_pre':'categorical_crossentropy',
            'gender_pre':'binary_crossentropy',
        },
        loss_weights={
            'age_pre':0.25,
            'income_pre':1.,
            'gender_pre':10.,
        },
    )
    model.fit(
        posts,
        {
            'age_pre': age_target,
            'income_pre': income_target,
            'gender_pre': gender_target,
        },
        epochs=10,
        batch_size=64,
    )
    '''

    # residual(size no same)
    '''
    x = ...
    y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
    y = layers.MaxPool2D(2, strides=2)(y)

    residual = layers.Conv2D(128, 1, strides=2, padding='same')(x)
    y = layers.add([y, residual])
    '''

    # layer weights shared
    '''
    lstm = layers.LSTM(32)

    left_input = Input(shape=(None, 128))
    left_output = lstm(left_input)
    right_input = Input(shape=(None, 128))
    right_output = lstm(left_input)

    merged = layers.concatenate([left_output, right_output], axis=-1)
    predictions = layers.Dense(1, activation='sigmoid')(merged)

    model = Model([left_output, right_output], predictions)
    model.fit([left_data, right_data], targets)
    '''

    # model as layer
    '''
    y = model(x)
    y1, y2 = model([x1, x2])
    '''

    # xception implementation
    '''
    xception_base = applications.Xception(weights=None, include_top=False)
    left_input = Input(shape=(250, 250, 3))
    right_input = Input(shape=(250, 250, 3))

    left_features = xception_base(left_input)
    right_input = xception_base(right_input)

    merged_features = layers.concatenate([left_features, right_input], axis=-1)
    '''

    # callback
    '''
    callbacks.ModelCheckpoint
    callbacks.EarlyStopping
    callbacks.TensorBoard
    callbacks.CSVLogger
    callbacks.ReduceLROnPlateau
    '''
    '''
    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath='model.h5',
            monitor='val_loss',
            save_best_only=True,
        ),
        callbacks.EarlyStopping(
            monitor='acc',
            patience=3,
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=3,
        ),
    ]
    model.fit(x, y,
        epochs=10,
        batch_size=32,
        callbacks=callbacks_list,
        validation_data=(x_val, y_val),
    )
    '''

    callbacks_list = [
        callbacks.TensorBoard(
            log_dir='.',
            histogram_freq=1,
            embeddings_freq=1,
        )
    ]
