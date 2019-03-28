import os
import glob
import json
import numpy as np
import pandas as pd
from keras.models import model_from_json
from keras import backend as K
from .utils import feature_extraction, latest_block


def main(device, dataset, model_architecture="double_lstm", model_dir="ml/outputs"):
    features, minima, maxima, scaling_parameter = feature_extraction(dataset)
    window = 5

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

    result = []
    for i in range(5):
        ip = latest_block(features[::-1], window)
        print(i, ip)
        predicted2 = model.predict(ip)
        result.append(((predicted2 * scaling_parameter) + minima).item(0))
        features = features.append(pd.DataFrame([[1, 1, 1, 1, features.tail(1)[4], predicted2]]))

    K.clear_session()

    return result
