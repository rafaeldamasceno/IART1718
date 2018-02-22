import tensorflow as tf
import pandas as pd

TRAIN_PATH = "HTRU_2.csv"

CSV_COLUMN_NAMES = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
CLASS = ['Pulsar', 'NotPulsar']

train = pd.read_csv(TRAIN_PATH, names=CSV_COLUMN_NAMES, header=0)
train_x, train_y = train, train.pop('9')

# Create feature columns for all features.
my_feature_columns = []
for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

# print(my_feature_columns)

classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[5],
    n_classes=2)

dataset = tf.data.Dataset.from_tensor_slices((dict(train_x), train_y))
dataset = dataset.shuffle(1000).repeat().batch(100)

classifier.train(input_fn=dataset, steps=1000)

eval_result = classifier.evaluate(input_fn=dataset, steps=1000)

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
