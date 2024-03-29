from keras.layers import BatchNormalization, Dense, Input
from keras.models import Sequential
from keras.utils.vis_utils import plot_model


def simple_mixture_model(inshape: int):
    model = Sequential()
    model.add(Input(shape=(inshape,)))

    model.add(Dense(inshape * 2, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(inshape * 4, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(inshape))

    model.compile(loss="mae", optimizer="adam")
    return model


def fit_model(model, train_X, train_y, test_X, test_y, batch_size: int):
    batch_size = batch_size
    validation_data = (test_X, test_y)

    # fit network
    history = model.fit(
        train_X,
        train_y,
        epochs=20,
        batch_size=batch_size,
        validation_data=validation_data,
        verbose=1,
    )

    return model, history
