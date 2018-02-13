# Imports
import tensorflow as tf


# Create the network
def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Features["x"] is the tensor we are passing and 28x28x1 are the dimensions of the image
    # -1 is used for setting the number of batches dynamically
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1 - Where's the stride?
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,                 # Number of filters
        kernel_size=[5, 5],         # Size of the filters
        padding="same",             # Zero-padding to preserve the input size
        activation=tf.nn.relu)      # ReLU activation function

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,               # The input is the output of Conv1
        pool_size=[2, 2],           # Size of the filters
        strides=2)                  # Movement of the stride

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Check this out!!
    dropout = tf.layers.dropout(
        inputs=dense,                                   # The input layer is dense
        rate=0.4,                                       # 40% of the elements will be randomly dropped out
        training=mode == tf.estimator.ModeKeys.TRAIN)   # Dropout will only be performed if training mode

    # Logits Layer - Give the probability for each class
    logits = tf.layers.dense(
        inputs=dropout,             # The input layer
        units=10)                   # There are 10 classes [0, 9]

    # Our output has shape [batch_size, 10]

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),     # Return the largest value across an axes of a tensor

        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
        "probabilities": tf.nn.softmax(
            logits,                     # We apply softmax to logits tensor
            name="softmax_tensor")      # It's the name of this op, we can refer to it later
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes) and return it as a tensor
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,                  # The correct class in the training data
        logits=logits)                  # The output of the CNN

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(
          mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
