import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

DATASET_PATH = "HTRU_2.csv"
TRAIN_SAMPLE = 0.8

CSV_COLUMN_NAMES = ['Profile_mean',
                    'Profile_stdev',
                    'Profile_skewness',
                    'Profile_kurtosis',
                    'DM_mean',
                    'DM_stdev',
                    'DM_skewness',
                    'DM_kurtosis',
                    'class']
CLASS = ['Negative', 'Positive']


def load_data(y_name='class', random=False, balance_weight=False, normalize=False):
    data = pd.read_csv(DATASET_PATH, names=CSV_COLUMN_NAMES, header=None)

    if normalize:
        norm = (data - data.mean()) / data.std()
        norm['class'] = data['class']
        data = norm

    if balance_weight:
        data = data.apply(np.random.permutation)
        negative_count = data.groupby('class').size()[0]
        positive_count = data.groupby('class').size()[1]
        for index, row in data.iterrows():
            if (not row['class']):
                data = data.drop(index)
                negative_count -= 1
            if (negative_count == positive_count * balance_weight):
                break

    train, test = train_test_split(data, test_size=1-TRAIN_SAMPLE, shuffle=random)

    train_x, train_y = train, train.pop(y_name)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(batch_size).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset