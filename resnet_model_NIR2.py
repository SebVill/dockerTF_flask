# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for the preactivation form of Residual Networks.

Residual networks (ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant implemented in this module was
introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer
rather than after.
"""



#RAPPEL: istraining=true dans en apprentissage, = False en test ou valid

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
BN_EPSILON = 0.001
weight_decay=0.0002

class Model(object):
    def __init__(self, batch_size=32, learning_rate=1e-3, num_labels=65, is_training=True):
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._num_labels = num_labels
        self.is_training = is_training

    def train(self, loss, global_step):
        train_op = tf.train.MomentumOptimizer(self._learning_rate, 0.9).minimize(loss, global_step=global_step)
        return train_op

    def loss(self, logits, labels):
        with tf.variable_scope('loss') as scope:
          cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
          cost = tf.reduce_mean(cross_entropy, name=scope.name)
          return cost

    def activation_summary(self, x):
        '''
        :param x: A Tensor
        :return: Add histogram summary and scalar summary of the sparsity of the tensor
        '''
        tensor_name = x.op.name
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


    def create_variables(self, name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
        '''
        :param name: A string. The name of the new variable
        :param shape: A list of dimensions
        :param initializer: User Xavier as default.
        :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
        layers.
        :return: The created variable
        '''

        ## TODO: to allow different weight decay to fully connected layer and conv layer
        if is_fc_layer is True:
            regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
        else:
            regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)

        new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                        regularizer=regularizer)
        return new_variables


    def output_layer(self, input_layer, num_labels):
        '''
        :param input_layer: 2D tensor
        :param num_labels: int. How many output labels in total? (10 for cifar10 and 100 for cifar100)
        :return: output layer Y = WX + B
        '''
        input_dim = input_layer.get_shape().as_list()[-1]
        fc_w = self.create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True,
                                initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        fc_b = self.create_variables(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer())

        fc_h = tf.matmul(input_layer, fc_w) + fc_b
        return fc_h

    '''
    def batch_normalization_layer(self, input_layer, dimension):
        #Helper function to do batch normalziation
        #:param input_layer: 4D tensor
        #:param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
        #:return: the 4D tensor after being normalized

        mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
        beta = tf.get_variable('beta', dimension, tf.float32,
                                   initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable('gamma', dimension, tf.float32,
                                    initializer=tf.constant_initializer(1.0, tf.float32))
        bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

        return bn_layer

    '''

    def batch_normalization_layer(self, x, phase_train, affine=True):
        """
        x
            Tensor, 4D BHWD input maps
        phase_train
            boolean tf.Variable, true indicates training phase
        scope
            string, variable scope
        affine
            whether to affine-transform outputs
        Return
        ------
        normed
            batch-normalized maps
        """
        shape = x.get_shape().as_list()

        beta = tf.Variable(tf.constant(0.0, shape=[shape[-1]]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[shape[-1]]),
                            name='gamma', trainable=affine)

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        ema_apply_op = ema.apply([batch_mean, batch_var])
        ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

        def mean_var_with_update():
            """Summary
            Returns
            -------
            name : TYPE
                Description
            """
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema_mean, ema_var))

        normed = tf.nn.batch_norm_with_global_normalization(x, mean, var, beta, gamma, 1e-3, affine)
        return normed

    def conv_bn_relu_layer(self, input_layer, filter_shape, stride):
        '''
        A helper function to conv, batch normalize and relu the input tensor sequentially
        :param input_layer: 4D tensor
        :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
        :param stride: stride size for conv
        :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
        '''

        out_channel = filter_shape[-1]
        filter = self.create_variables(name='conv', shape=filter_shape)

        conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
        bn_layer = self.batch_normalization_layer(conv_layer, self.is_training)

        output = tf.nn.relu(bn_layer)
        return output


    def bn_relu_conv_layer(self, input_layer, filter_shape, stride):
        '''
        A helper function to batch normalize, relu and conv the input layer sequentially
        :param input_layer: 4D tensor
        :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
        :param stride: stride size for conv
        :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
        '''

        in_channel = input_layer.get_shape().as_list()[-1]

        bn_layer = self.batch_normalization_layer(input_layer, self.is_training)
        relu_layer = tf.nn.relu(bn_layer)

        filter = self.create_variables(name='conv', shape=filter_shape)
        conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
        return conv_layer



    def residual_block(self, input_layer, output_channel, first_block=False):
        '''
        Defines a residual block in ResNet
        :param input_layer: 4D tensor
        :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
        :param first_block: if this is the first residual block of the whole network
        :return: 4D tensor.
        '''
        input_channel = input_layer.get_shape().as_list()[-1]

        # When it's time to "shrink" the image size, we use stride = 2
        if input_channel * 2 == output_channel:
            increase_dim = True
            stride = 2
        elif input_channel == output_channel:
            increase_dim = False
            stride = 1
        else:
            raise ValueError('Output and input channel does not match in residual blocks!!!')

        # The first conv layer of the first residual block does not need to be normalized and relu-ed.
        with tf.variable_scope('conv1_in_block'):
            if first_block:
                filter = self.create_variables(name='conv', shape=[3, 3, input_channel, output_channel])
                conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
            else:
                conv1 = self.bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride)

        with tf.variable_scope('conv2_in_block'):
            conv2 = self.bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1)

        # When the channels of input layer and conv2 does not match, we add zero pads to increase the
        #  depth of input layers
        if increase_dim is True:
            pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='VALID')
            padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                         input_channel // 2]])
        else:
            padded_input = input_layer

        output = conv2 + padded_input
        return output


    def inference(self, input_tensor_batch, n, reuse):
        '''
        The main function that defines the ResNet. total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
        :param input_tensor_batch: 4D tensor
        :param n: num_residual_blocks
        :param reuse: To build train graph, reuse=False. To build validation graph and share weights
        with train graph, resue=True
        :return: last layer in the network. Not softmax-ed
        '''

        layers = []
        with tf.variable_scope('conv0', reuse=reuse):
            conv0 = self.conv_bn_relu_layer(input_tensor_batch, [3, 3, 3, 16], 1)
            self.activation_summary(conv0)
            layers.append(conv0)

        for i in range(n):
            with tf.variable_scope('conv1_%d' %i, reuse=reuse):
                if i == 0:
                    conv1 = self.residual_block(layers[-1], 16, first_block=True)

                else:
                    conv1 = self.residual_block(layers[-1], 16)

                self.activation_summary(conv1)
                layers.append(conv1)

        for i in range(n):
            with tf.variable_scope('conv2_%d' %i, reuse=reuse):
                conv2 = self.residual_block(layers[-1], 32)

                self.activation_summary(conv2)
                layers.append(conv2)

        for i in range(n):
            with tf.variable_scope('conv3_%d' %i, reuse=reuse):
                conv3 = self.residual_block(layers[-1], 64)
                layers.append(conv3)
            #assert conv3.get_shape().as_list()[1:] == [32, 32, 64]#128
            assert conv3.get_shape().as_list()[1:] == [16, 16, 64]#64

        with tf.variable_scope('fc', reuse=reuse):
            in_channel = layers[-1].get_shape().as_list()[-1]
            bn_layer = self.batch_normalization_layer(layers[-1], self.is_training)
            relu_layer = tf.nn.relu(bn_layer)
            global_pool = tf.reduce_mean(relu_layer, [1, 2])

            assert global_pool.get_shape().as_list()[-1:] == [64]
            output = self.output_layer(global_pool, self._num_labels)
            layers.append(output)

        return layers[-1]

    def accuracy(self, logits, y):
        with tf.variable_scope('accuracy') as scope:
          accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(logits, 1), dtype=tf.int64), y), dtype=tf.float32),name=scope.name)
          tf.summary.scalar('accuracy', accuracy)
          return accuracy

    def predictions(self, logits):
		with tf.variable_scope('predictions') as scope:
			predictions=tf.nn.softmax(logits, name='pred')
			tf.summary.scalar('predictions', predictions)
		return predictions
