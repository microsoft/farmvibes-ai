from keras import regularizers
from keras.layers import (
    LSTM,
    BatchNormalization,
    Dense,
    Flatten,
    Input,
    LocallyConnected1D,
    RepeatVector,
    TimeDistributed,
    concatenate,
)
from keras.models import Model
from keras.optimizers import Adam
from matplotlib import pyplot as plt

from .transformer_models_ts import Encoder


def moddeepmc_pred_model(train_X, train_y):

    n_outputs = train_y.shape[1]

    inputs = list()

    k = 0
    kernel_size = [2, 2, 2, 2, 2, 2, 2]

    t_in = Input(
        shape=(
            train_X[k].shape[1],
            train_X[k].shape[2],
        )
    )
    conv1 = LocallyConnected1D(
        3, kernel_size[k], strides=1, activation="relu", kernel_initializer="he_normal"
    )(t_in)
    conv1 = BatchNormalization()(conv1)

    transformer = hidden_transformer_layer(
        conv1,
        num_layers=2,
        d_model=4,
        num_heads=4,
        dff=16,
        pe_input=conv1.shape[1],
        rate=0.1,
    )

    inputs.append(t_in)

    flat_tfmr = Flatten()(transformer)

    k = k + 1

    flat = [flat_tfmr]
    for i in range(k, len(train_X)):
        t_in, t_flat = cnnlstm_layers(train_X[k], kernel_size[k])
        inputs.append(t_in)
        flat.append(t_flat)
        k = k + 1

    merge = concatenate(flat)
    merge = BatchNormalization()(merge)

    # Decoder
    repeat1 = RepeatVector(n_outputs)(merge)

    lstm2 = LSTM(
        20,
        activation="relu",
        return_sequences=True,
        kernel_regularizer=regularizers.l1_l2(l1=1e-3, l2=1e-3),
        activity_regularizer=regularizers.l2(l2=1e-3),
    )(repeat1)

    lstm2 = BatchNormalization()(lstm2)
    dense1 = TimeDistributed(Dense(16, activation="relu", kernel_regularizer=None))(lstm2)
    output = TimeDistributed(Dense(1))(dense1)

    # Model Creation
    model = Model(inputs=inputs, outputs=output)

    opt = Adam(learning_rate=0.002)
    model.compile(loss="mse", optimizer=opt)
    return model


def deepmc_fit_model(
    model: Model,
    train_X,
    train_y,
    validation_data=None,
    server_mode: bool = False,
    epochs: int = 30,
):
    batch_size = 8

    # fit network
    history = model.fit(
        train_X,
        train_y,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        verbose=1,
    )

    # plot history
    if server_mode is False:
        plt.plot(history.history["loss"], label="train")
        if validation_data is not None:
            plt.plot(history.history["val_loss"], label="test")

        plt.legend()
        plt.show()

    return model, history


def cnnlstm_layers(train_X, kernel_size: int = 4):
    # design network
    n_timesteps, n_features = train_X.shape[1], train_X.shape[2]
    in1 = Input(
        shape=(
            n_timesteps,
            n_features,
        )
    )
    kernel_regularizer = None  # regularizers.l1_l2(l1=1e-4, l2=1e-4)
    conv1 = LocallyConnected1D(
        4,
        kernel_size,
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=kernel_regularizer,
    )(in1)
    conv1 = BatchNormalization()(conv1)
    conv2 = LocallyConnected1D(
        8,
        kernel_size,
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=kernel_regularizer,
    )(conv1)
    conv2 = BatchNormalization()(conv2)

    lstm1 = LSTM(
        16,
        activation="relu",
        return_sequences=False,
        recurrent_dropout=0.2,
        dropout=0.2,
        kernel_regularizer=kernel_regularizer,
    )(conv2)

    return in1, lstm1


def hidden_transformer_layer(
    layer,
    num_layers: int,
    d_model: int,
    num_heads: int,
    dff: int,
    pe_input: int,
    rate: float,
):
    encoder = Encoder(num_layers, d_model, num_heads, dff, pe_input, rate)
    encoded = encoder(layer, True, None)
    return encoded
