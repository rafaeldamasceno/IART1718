from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

from math import sqrt

import pulsar_data

parser = argparse.ArgumentParser()
parser.add_argument('--layers', default=3, type=int, help='number of intermediate layers')
parser.add_argument('--nodes', default=12, type=int,
                    help='number of nodes per layer')
parser.add_argument('--random', default=False, type=bool, help='randomize dataset')
parser.add_argument('--balance_weight', default=0, type=float, help='balance weight')
parser.add_argument('--normalize', default=False, type=bool, help='normalize dataset')

def my_model(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(argv):

    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = pulsar_data.load_data(random=args.random, balance_weight=args.balance_weight, normalize=args.normalize)

    TRAIN_STEPS = train_x.shape[0]
    BATCH_SIZE= int(TRAIN_STEPS * 0.1)

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params={
            'feature_columns': my_feature_columns,
            'hidden_units': args.layers * [args.nodes],
            'n_classes': 2,
        })

    # Train the Model.
    classifier.train(
        input_fn=lambda: pulsar_data.train_input_fn(
            train_x, train_y, BATCH_SIZE),
        steps=TRAIN_STEPS)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda: pulsar_data.eval_input_fn(test_x, test_y, BATCH_SIZE))

    print('\nTest set accuracy: {accuracy:f}\n'.format(**eval_result))

    # Generate predictions from the model
    expected = ['Negative', 'Positive']
    predict_x = {
        'Profile_mean': [141.1875, 99.3671875],
        'Profile_stdev': [39.60937192, 41.57220208],
        'Profile_skewness': [-0.172315843, 1.547196967],
        'Profile_kurtosis': [0.997104608, 4.154106043],
        'DM_mean': [2.731605351, 27.55518395],
        'DM_stdev': [19.39785108, 61.71901588],
        'DM_skewness': [8.826834558, 2.20880796],
        'DM_kurtosis': [85.66471835, 3.662680136]
    }

    predictions = classifier.predict(
        input_fn=lambda: pulsar_data.eval_input_fn(predict_x,
                                                   labels=None,
                                                   batch_size=BATCH_SIZE))

    for pred_dict, expec in zip(predictions, expected):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(pulsar_data.CLASS[class_id],
                              100 * probability, expec))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)