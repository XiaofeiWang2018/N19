
import tensorflow as tf
import math
# import numpy as np
# from config import Config




class BasicNet(object):
    weight_decay = 5*1e-6
    bias_conv_init = 0.1 #weight init for biasis
    bias_fc_init = 0.1
    leaky_alpha = 0.1
    is_training = False
    #LL_VARIABLES = 'll_variables'

    #batch_normalization = True
    # class_num = 2

    def _get_variable(self,
                      name,
                      shape,
                      initializer,
                      weight_decay= weight_decay,
                      dtype='float32',
                      trainable=True, AAAI_VARIABLES=None):  # pretrain/ initial/

        if weight_decay >0:
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        else:
            regularizer = None

        collection = [tf.GraphKeys.GLOBAL_VARIABLES]  #, LL_VARIABLES

        return tf.get_variable(name= name,
                               shape= shape,
                               initializer= initializer,
                               regularizer=regularizer,
                               collections= collection,
                               dtype= dtype,
                               trainable= trainable,
                               )

    def conv(self, scope_name, x, ksize, filters_out, stride=1, batch_norm= True, is_training= True, liner = False, reuse=None):
        with tf.variable_scope(scope_name, reuse=reuse):
            filters_in = x.get_shape()[-1].value

            shape = [ksize, ksize, filters_in, filters_out]  # conv kernel size
            weights = self._get_variable('weights',
                                    shape=shape,
                                    initializer=tf.contrib.layers.xavier_initializer()  # need to set seed number
                                    )
            bias = self._get_variable('bias',
                                 shape=[filters_out],
                                 initializer=tf.constant_initializer(self.bias_conv_init)
                                 )
            conv = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')
            conv_bias = tf.nn.bias_add(conv, bias, name='linearout')

            if batch_norm:
                out = self.bn(conv_bias, is_training=is_training)
            else:
                out = conv_bias

            if liner:
                return out
            else:
                return self.leaky_relu(out)

    def dconv(self, scope_name, x, ksize, filters_out, stride=1, liner=False):
        with tf.variable_scope(scope_name):
            filters_in = x.get_shape()[-1].value

            shape = [ksize, ksize, filters_out, filters_in]
            weights = self._get_variable('weights',
                                    shape=shape,
                                    initializer=tf.contrib.layers.xavier_initializer()
                                    )
            bias = self._get_variable('bias',
                                 shape=[filters_out],
                                 initializer=tf.constant_initializer(self.bias_conv_init)
                                 )

            output_shape = tf.stack([tf.shape(x)[0], tf.shape(x)[1] * stride, tf.shape(x)[2] * stride, filters_out])
            conv = tf.nn.conv2d_transpose(x, weights, output_shape, strides=[1, stride, stride, 1], padding='SAME')
            conv_biased = tf.nn.bias_add(conv, bias, name='linearout')

            if liner:
                return conv_biased
            else:
                return self.leaky_relu(conv_biased)

    def fc(self, scope_name, x, class_num, flat=False, linear=False):
        with tf.variable_scope(scope_name):
            input_shape = x.get_shape().as_list()
            if flat:
                dim = input_shape[1] * input_shape[2] * input_shape[3]
                input_processed = tf.reshape(x, [-1, dim])  # 2[batch, feature]
            else:
                dim = input_shape[1]  # already flat 2 [batch, hidden_feature]
                input_processed = x

            weights = self._get_variable(name='weights',
                                         shape=[dim, class_num],
                                         initializer=tf.contrib.layers.xavier_initializer()
                                    )
            bias = self._get_variable(name='bias',
                                      shape=[class_num],
                                      initializer=tf.constant_initializer(self.bias_fc_init))

            out = tf.add(tf.matmul(input_processed, weights), bias, name='linearout')  # [batch, class_num]

            if linear:
                return out
            else:
                return self.leaky_relu(out)

    def bn(self, x, is_training):
        return tf.layers.batch_normalization(x, training=is_training)

    def max_pool(self, x, ksize=3, stride=2):
        return tf.nn.max_pool(x,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, stride, stride, 1],
                              padding='SAME')

    def leaky_relu(self, x, leaky_alpha=leaky_alpha, dtype=tf.float32):
        x = tf.cast(x, dtype=dtype)
        bool_mask = (x > 0)
        mask = tf.cast(bool_mask, dtype=dtype)
        return 1.0 * mask * x + leaky_alpha * (1 - mask) * x

    def bottleneck(self, x, stride):    #stride

        input_channel=x.get_shape()[-1]
        output_channel = 4 * input_channel

        shortcut = x
        #with tf.variable_scope('a'):
        x = self.conv('a', x, ksize=1, filters_out=input_channel, stride=stride)
        #with tf.variable_scope('b'):
        x = self.conv('b', x, ksize=3, filters_out=input_channel, stride=1)
        #with tf.variable_scope('c'):
        x = self.conv('c', x, ksize=1, filters_out=output_channel, stride=1, liner=True)
        #with tf.variable_scope('shortcut'):
        shortcut = self.conv('shortcut', shortcut, ksize=1, filters_out=output_channel, stride=stride, liner=True)

        return self.leaky_relu(x + shortcut)

    def building_block(self, scope_name, x, output_channel, stride=2, reuse=None):
        with tf.variable_scope(scope_name):
            input_channel = x.get_shape()[-1]
            # output_channel = input_shape

            shortcut = x
            #with tf.variable_scope('A'):
            x = self.conv('A', x, ksize=3, filters_out=input_channel, stride=stride, reuse=reuse)
            #with tf.variable_scope('B'):
            x = self.conv('B', x, ksize=3, filters_out=output_channel, stride=1, liner=True, reuse=reuse)
            #with tf.variable_scope('Shortcut'):
            if output_channel != input_channel or stride != 1:
                shortcut = self.conv('Shortcut', shortcut, ksize=1, filters_out=output_channel, stride=stride, liner=True, reuse=reuse)

            return self.leaky_relu(x + shortcut)

    def conv_mask(self, x, mask):
        tempsize = x.get_shape().as_list()
        mask_resize = tf.image.resize_images(mask, [tempsize[1], tempsize[2]])

        return x * mask_resize

    def _normlized_0to1(self, mat): # tensor [batch_size, image_height, image_width, channels] normalize each fea map(??salency map??)
        mat_shape = mat.get_shape().as_list()
        tempmin = tf.reduce_min(mat, axis=1)
        tempmin= tf.reduce_min(tempmin, axis=1)     #each batch,each channel , the minimize of each salency map,[batch,1]  [[0.1],[0.05]...,[0.02]]
        tempmin = tf.reshape(tempmin, [-1, 1, 1, mat_shape[3]])
        tempmat = mat - tempmin     # for min=0
        tempmax = tf.reduce_max(tempmat, axis=1)
        tempmax = tf.reduce_max(tempmax, axis=1) + self.eps
        tempmax = tf.reshape(tempmax, [-1, 1, 1, mat_shape[3]])

        return tempmat / tempmax

    def _normlized(self, mat):  # tensor [batch_size, image_height, image_width, channels] normalize each fea map,  max_value to
        mat_shape = mat.get_shape().as_list()
        tempsum = tf.reduce_sum(mat, axis=1)
        tempsum = tf.reduce_sum(tempsum, axis=1) + self.eps          #each batch,each channel have a value,sum of each feature map(w*h) [batch_size, channel]
        tempsum = tf.reshape(tempsum, [-1, 1, 1, mat_shape[3]])
        return mat / tempsum

    # def lstm(self, scope_name, rnn_size, layer_depth):
    #     with tf.variable_scope(scope_name):
    #         cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)  # rnn_size=650
    #         stacked_cell = tf.nn.rnn_cell.MultiRNNCell([cell] * layer_depth)  # layer_depth=2
	#
    #     return

    def lstm(self, scope_name, x, n_layers, n_neurons, n_outputs, sequence_length, state_is_tuple=True, state_is_zero_init=True):

        """
        :param x: x_size = [None, n_steps, n_inputs]
        :param n_layers: layers of lstm. to stack
        :param n_neurons: size of cell in lstm
        :param n_outputs: dense output size
                state_is_tuple: structure type of state in lstm cells
                state_is_zero_init : the way to initialize the lstm cell
                n_steps: sequence length, in batch, size = [batch_size]
        :return:rnn_output_size=(batch, n_steps, n_neurons)
                fc_output_size=(batch, n_steps, n_outputs)
                state: The final state


        state_is_tuple=False
        state_init
        size = [batch_size, n_layers * 2 * n_neurons]
        state_is_tuple=True  
        size = (LSTMStateTuple(c=<tensor, size=[batch_size, n_neurons]>,
        					   h=<tensor, size=[batch_size, n_neurons]>),) *n_layers
        """
        with tf.variable_scope(scope_name):
            cells = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, state_is_tuple=state_is_tuple) for _ in
                     range(n_layers)]  # state_is_tuple decide the size of states
            cells = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=0.5) for cell in cells]
            cells = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=state_is_tuple)

            batch_size = x.get_shape()[0]       # .as_list()
            n_steps = x.get_shape()[1]
            n_input = x.get_shape()[-1]

            if state_is_zero_init:
                if state_is_tuple:
                    lstm_state_c = tf.zeros(shape=(batch_size, n_neurons))
                    lstm_state_h = tf.zeros(shape=(batch_size, n_neurons))
                    lstm_state = tf.nn.rnn_cell.LSTMStateTuple(c=lstm_state_c, h=lstm_state_h)
                    state_init = (lstm_state,) * n_layers
                else:
                    state_init = tf.zeros([batch_size, n_layers * 2 * n_neurons])
                rnn_outputs, states = tf.nn.dynamic_rnn(cells, x, dtype=tf.float32, initial_state=state_init)
            else:
                rnn_outputs, states = tf.nn.dynamic_rnn(cells, x, sequence_length=sequence_length, dtype=tf.float32)    #sequence_length=n_steps,

            stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
            stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
            fc_output = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs], name="logits")  # 预测结果

            return rnn_outputs, fc_output, states


    """calculate_acc_sensitivity_specificity in one batch"""
    def _result(self, labGTs, predictions):     #calculate on each validation batch       #label_predict_op=(batch, n_steps, n_outputs)
        e = math.e
        #print(labGTs)
        #print(predictions)
        batch_size_1 = labGTs.shape[0]      #ndarray
        t = 0
        # tp = 0
        # tn = 0
        # fp = 0
        # fn = 0
        n_steps1 = labGTs.shape[1]
        n_steps2 = predictions.shape[1]
        assert n_steps1 == n_steps2
        for i in range(batch_size_1):
            for j in range(n_steps1):
                prediction_0 = e ** (predictions[i, j, 0]) / (e ** (predictions[i, j, 1]) + e ** (predictions[i, j, 0]) )
                prediction_1 = e ** (predictions[i, j, 1]) / (e ** (predictions[i, j, 1]) + e ** (predictions[i, j, 0]) )
               # prediction_2 = e ** (predictions[i, j, 2]) / (e ** (predictions[i, j, 1]) + e ** (predictions[i, j, 0]) + e ** (predictions[i, j, 2]))
                # print('prediction_0', prediction_0, 'prediction_1', prediction_1, 'sum', prediction_0 + prediction_1)
                #print('label', labGTs[i])
                #assert prediction_0 + prediction_1 == 1
                assert labGTs[i, j] == 1 or labGTs[i, j] == 0# or labGTs[i, j] == 2
                if labGTs[i, j] == 1:
                    if prediction_1 >= prediction_0: #and prediction_1 >= prediction_2:
                        t = t + 1
                elif labGTs[i, j] == 0:
                    if prediction_0 >= prediction_1: #and prediction_0 >= prediction_2:
                        t = t + 1
                # elif labGTs[i, j] == 2:
                #     if prediction_2 >= prediction_1 and prediction_2 >= prediction_0:
                #         t = t + 1
        # assert (tp + tn + fp + fn) == batch_size_1 * n_steps1
        acc = t / (batch_size_1 * n_steps1)
        return acc