import tensorflow as tf


class LrModel(object):
    def __init__(self, config, seq_length):
        self.config = config
        self.seq_length = seq_length
        self.lr()

    def lr(self):
        self.x = tf.placeholder(tf.float32, [None, self.seq_length])
        w = tf.Variable(tf.zeros([self.seq_length, self.config.num_classes]))
        b = tf.Variable(tf.zeros([self.config.num_classes]))

        y = tf.nn.softmax(tf.matmul(self.x, w) + b)

        self.y_pred_cls = tf.argmax(y, 1)

        self.y_ = tf.placeholder(tf.float32, [None, self.config.num_classes])
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(y), reduction_indices=[1]))
        self.loss = tf.reduce_mean(cross_entropy)

        self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))