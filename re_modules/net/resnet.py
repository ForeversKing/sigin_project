from collections import namedtuple
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.training import moving_averages

HParams = namedtuple('HParams',
                     'batch_size, num_residual_units, use_bottleneck, '
                     'relu_leakiness')


class ResNet(object):
    """ResNet model."""

    def __init__(self, hps, images, is_training, reuse=None):
        """ResNet constructor.

        Args:
          hps: Hyperparameters.
          images: Batches of images. [batch_size, image_size, image_size, 3]
          labels: Batches of labels. [batch_size, num_classes]
          mode: One of 'train' and 'eval'.
        """
        self.hps = hps
        self._images = images
        self.is_training = is_training
        self.reuse = reuse
        self._extra_train_ops = []

    @staticmethod
    def _stride_arr(stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def build_model(self):
        """Build the core model within the graph."""
        strides = [1, 2, 2, 2]
        activate_before_residual = [True, False, False, False]
        if self.hps.use_bottleneck:
            res_func = self._bottleneck_residual
            filters = [16, 64, 128, 256, 512]
        else:
            res_func = self._residual
            # filters = [16, 16, 32, 64, 128]
            filters = [64, 64, 128, 256, 512]

        with tf.variable_scope('init', reuse=self.reuse):
            x = self._images
            x = self._conv('init_conv', x, 3, 3, filters[0], self._stride_arr(1))

        with tf.variable_scope('unit_1_0', reuse=self.reuse):
            x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                         activate_before_residual[0])
        for i in range(1, self.hps.num_residual_units):
            with tf.variable_scope('unit_1_%d' % i, reuse=self.reuse):
                x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)
        print('residual 1: ', x.shape)

        with tf.variable_scope('unit_2_0', reuse=self.reuse):
            x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                         activate_before_residual[1])
        for i in range(1, self.hps.num_residual_units):
            with tf.variable_scope('unit_2_%d' % i, reuse=self.reuse):
                x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)
        print('residual 2: ', x.shape)

        with tf.variable_scope('unit_3_0', reuse=self.reuse):
            x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                         activate_before_residual[2])
        for i in range(1, self.hps.num_residual_units):
            with tf.variable_scope('unit_3_%d' % i, reuse=self.reuse):
                x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)
        print('residual 3: ', x.shape)

        with tf.variable_scope('unit_4_0', reuse=self.reuse):
            x = res_func(x, filters[3], filters[4], self._stride_arr(strides[3]),
                         activate_before_residual[3])
        for i in range(1, self.hps.num_residual_units):
            with tf.variable_scope('unit_4_%d' % i, reuse=self.reuse):
                x = res_func(x, filters[4], filters[4], self._stride_arr(1), False)
        print('residual 4: ', x.shape)

        with tf.variable_scope('unit_last', reuse=self.reuse):
            x = self._batch_norm('final_bn', x)
            x = self._relu(x, self.hps.relu_leakiness)
            # x = self._reshape_3dim(x, filters[-1])
        print('residual 5: ', x.shape)
        return x

    def _batch_norm(self, name, x):
        return tf.layers.batch_normalization(x, axis=3,  # channels last,
                                             momentum=0.9,
                                             epsilon=2e-5,
                                             training=self.is_training,
                                             trainable=self.is_training,
                                             name=name)

    # TODO(xpan): Consider batch_norm in contrib/layers/python/layers/layers.py
    def _batch_norm_deprecated(self, name, x):
        """Batch normalization."""
        with tf.variable_scope(name, self.reuse):
            params_shape = [x.get_shape()[-1]]

            beta = tf.get_variable(
                'beta', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(
                'gamma', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32))

            if self.mode == 'train':
                mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

                moving_mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                moving_variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)

                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_mean, mean, 0.9))
                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_variance, variance, 0.9))
            else:
                mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)
                tf.summary.histogram(mean.op.name, mean)
                tf.summary.histogram(variance.op.name, variance)
            # epsilon used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
            y = tf.nn.batch_normalization(
                x, mean, variance, beta, gamma, 0.001)
            y.set_shape(x.get_shape())
            return y

    def _residual(self, x, in_filter, out_filter, stride,
                  activate_before_residual=False):
        """Residual unit with 2 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('shared_activation', reuse=self.reuse):
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.hps.relu_leakiness)
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation', reuse=self.reuse):
                orig_x = x
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.hps.relu_leakiness)

        with tf.variable_scope('sub1', reuse=self.reuse):
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

        with tf.variable_scope('sub2', reuse=self.reuse):
            x = self._batch_norm('bn2', x)
            x = self._relu(x, self.hps.relu_leakiness)
            x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add', reuse=self.reuse):
            if in_filter != out_filter:
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
                orig_x = tf.pad(
                    orig_x, [[0, 0], [0, 0], [0, 0],
                             [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]])
            x += orig_x

        tf.logging.debug('image after unit %s', x.get_shape())
        return x

    def _bottleneck_residual(self, x, in_filter, out_filter, stride,
                             activate_before_residual=False):
        """Bottleneck residual unit with 3 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('common_bn_relu', reuse=self.reuse):
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.hps.relu_leakiness)
                orig_x = x
        else:
            with tf.variable_scope('residual_bn_relu', reuse=self.reuse):
                orig_x = x
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.hps.relu_leakiness)

        with tf.variable_scope('sub1', reuse=self.reuse):
            x = self._conv('conv1', x, 1, in_filter, out_filter / 4, stride)

        with tf.variable_scope('sub2', reuse=self.reuse):
            x = self._batch_norm('bn2', x)
            x = self._relu(x, self.hps.relu_leakiness)
            x = self._conv('conv2', x, 3, out_filter / 4, out_filter / 4, [1, 1, 1, 1])

        with tf.variable_scope('sub3', reuse=self.reuse):
            x = self._batch_norm('bn3', x)
            x = self._relu(x, self.hps.relu_leakiness)
            x = self._conv('conv3', x, 1, out_filter / 4, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add', reuse=self.reuse):
            if in_filter != out_filter:
                orig_x = self._conv('project', orig_x, 1, in_filter, out_filter, stride)
            x += orig_x

        tf.logging.info('image after unit %s', x.get_shape())
        return x

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'DW') > 0:
                costs.append(tf.nn.l2_loss(var))
                # tf.summary.histogram(var.op.name, var)

        return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name, reuse=self.reuse):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(
                'DW', [filter_size, filter_size, in_filters, out_filters],
                tf.float32, initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(2.0 / n)))
            return tf.nn.conv2d(x, kernel, strides, padding='SAME')

    def _relu(self, x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])

    def _reshape_3dim(self, x, filter_num):
        x = tf.reshape(x, (self.hps.batch_size, -1))
        return x


def get_resnet(images, is_training, batch_size, num_class, reuse=None):
    hps = HParams(batch_size=batch_size,
                  num_residual_units=2,
                  use_bottleneck=False,
                  relu_leakiness=0.0)
    print('shape', images.shape)
    model = ResNet(hps, images, is_training, reuse=reuse)
    x = model.build_model()
    flatten = slim.flatten(x, scope='flatten')
    print('!!!!!', flatten.shape)
    initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
    fc6 = slim.fully_connected(flatten, 1024, scope='fc6')
    if is_training:
        fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True,
                           scope='dropout6')
    cls_score = slim.fully_connected(fc6, num_class, weights_initializer=initializer,
                                       trainable=is_training,
                                       activation_fn=None, scope='cls_score')
    print('cls_score', cls_score.shape)
    return cls_score


if __name__ == '__main__':
    get_resnet(tf.placeholder(tf.float32, (1, 32, 320, 3)),
               tf.placeholder(tf.int32, (32, None)),
               True,
               1)
