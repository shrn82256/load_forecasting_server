import os
import glob
import pandas as pd
from keras.models import model_from_json
from keras import backend as K
from .utils import feature_extraction, latest_block


def main(device, dataset, model_architecture="triple_lstm", model_dir="ml/outputs"):
    window = 5
    features, minima, maxima, scaling_parameter = feature_extraction(dataset)

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
        predicted2 = model.predict(ip)
        # print("ip", ip)
        result.append(((predicted2 * scaling_parameter) + minima).item(0))
        features = features.append(pd.DataFrame([[ip[0][0][0], ip[0][0][1], predicted2[0][0]]]))

    K.clear_session()

    return result
