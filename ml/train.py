import os
import json
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.callbacks import LambdaCallback
from keras import backend as K
from flask import Blueprint
from .utils import feature_extraction, split_features


def build_single_lstm(layers):
    model = Sequential()
    model.add(LSTM(50, input_shape=(
        layers[1], layers[0]), return_sequences=False))
    model.add(Dense(1, activation="relu", kernel_initializer="uniform"))
    model.compile(loss="mse", optimizer="rmsprop", metrics=["accuracy"])
    return model


def build_double_lstm(layers):
    dropout = 0.2
    model = Sequential()
    model.add(LSTM(128, input_shape=(
        layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(64, input_shape=(
        layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(16, activation="relu", kernel_initializer="uniform"))
    model.add(Dense(1, activation="relu", kernel_initializer="uniform"))
    model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
    return model


def build_triple_lstm(layers):
    dropout = 0.2
    model = Sequential()
    model.add(LSTM(128, input_shape=(
        layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(64, input_shape=(
        layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(32, input_shape=(
        layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(16, activation="relu", kernel_initializer="uniform"))
    model.add(Dense(1, activation="relu", kernel_initializer="uniform"))
    model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
    return model


model_architectures = {
    # {"mape": 5.3233211026637965, "mse": 1054.0611588281695}
    "single_lstm": build_single_lstm,

    # {"mape": 5.311654460311499, "mse": 693.2166513376623}
    "double_lstm": build_double_lstm,

    # {"mape": 5.299372785959865, "mse": 702.1735391393771}
    "triple_lstm": build_triple_lstm,
}


def main(device, dataset, epochs=50, batch_size=1024, validation_split=0.2, model_architecture="double_lstm", output_dir="ml/outputs"):
    features, minima, maxima, scaling_parameter = feature_extraction(dataset)
    window = 5
    X_train, y_train, X_test, y_test = split_features(features[::-1], window)

    json_logging_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: print(json.dumps({
            "epoch": epoch,
            "loss": logs["loss"],
            "acc": logs["acc"],
            "val_loss": logs["val_loss"],
            "val_acc": logs["val_acc"],
        })),
    )

    # figure out which model architecture to use
    arch = model_architecture
    assert arch in model_architectures, "Unknown model architecture '%s'." % arch
    builder = model_architectures[arch]

    # build and train the model
    model = builder([len(features.columns), window, 1])
    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=[json_logging_callback],
        verbose=0)

    model_prefix = device + "-" + model_architecture + "-model-"

    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(output_dir, model_prefix + "layout.json"), "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(os.path.join(output_dir, model_prefix + "weights.h5"))

    K.clear_session()

    return True;
