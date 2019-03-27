import os
import glob
import json
import numpy as np
from math import sqrt
from keras.models import model_from_json
from keras import backend as K
from sklearn.metrics import mean_absolute_error
import argparse
from .utils import feature_extraction, split_features


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def main(device, dataset, model_architecture="double_lstm", model_dir="ml/outputs"):
    features, minima, maxima, scaling_parameter = feature_extraction(dataset)
    window = 5
    X_train, y_train, X_test, y_test = split_features(features[::-1], window)

    model_prefix = device + "-" + model_architecture + "-model-"

    # load json and create model
    layout_path = glob.glob(os.path.join(
        model_dir, model_prefix + "layout.json"))[0]
    json_file = open(layout_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # load weights into new model
    weights_path = glob.glob(os.path.join(
        model_dir, model_prefix + "weights.h5"))[0]
    model.load_weights(weights_path)

    predicted2 = model.predict(X_test)
    actual = y_test
    predicted2 = (predicted2 * scaling_parameter) + minima
    actual = (actual * scaling_parameter) + minima

    K.clear_session()

    mape2 = sqrt(mean_absolute_percentage_error(predicted2, actual))
    mse2 = mean_absolute_error(actual, predicted2)

    return {
        "mape": mape2,
        "mse": mse2
    }
