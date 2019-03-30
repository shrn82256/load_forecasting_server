import pandas as pd
import numpy as np


def get_dataframe(dataset):
    my_data = pd.read_csv(dataset, error_bad_lines=False)
    df = pd.DataFrame(my_data)
    return df


def feature_extraction(dataset):
    df = get_dataframe(dataset)

    values = df.values
    minima = np.amin(values[:, -1])
    maxima = np.amax(values[:, -1])
    scaling_parameter = maxima - minima

    for i in range(3):
        values[:, i] = (values[:, i] - np.amin(values[:, i])) / \
            (np.amax(values[:, i]) - np.amin(values[:, i]))

    df = pd.DataFrame(values)
    return df, minima, maxima, scaling_parameter


def split_features(features_data_frame, seq_len):
    amount_of_features = len(features_data_frame.columns)
    data = features_data_frame.as_matrix()
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    result = np.array(result)
    row = round(0.8 * result.shape[0])
    train = result[:int(row), :]
    x_train = train[:, :-1]
    y_train = train[:, -1][:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:, -1]
    x_train = np.reshape(
        x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(
        x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))
    return [x_train, y_train, x_test, y_test]


def latest_block(features_data_frame, seq_len):
    amount_of_features = len(features_data_frame.columns)
    data = features_data_frame.as_matrix()
    sequence_length = seq_len + 1
    result = [data[:sequence_length]]
    block = np.array(result)[:, :-1]
    return np.reshape(block, (block.shape[0], block.shape[1], amount_of_features))
