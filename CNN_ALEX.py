
import BasicNet
import tensorflow as tf

class Net(BasicNet.BasicNet):

    def __init__(self, batch_size=32, n_steps=16, n_layers=2, n_neurons=256, n_outputs=2,
                 init_lr=10 ** (-4), #max_step=15,
    ):
        super(Net, self).__init__()

        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs
        #self.max_step = max_step

        self.init_lr = init_lr

        self.predict_label = []
        self.loss = []
        self.loss_weight = []
        self.train = []

    def inference(self, x, is_training=True):         # x= [batch, n_step, w, h, c]
        #cnn_single = self.CNN('cnn_sig', x)
        cnn_sequential = self.multi_CNN_output_1D('cnn_mul', x, is_training)     #cnn_sequential = [batch, max_step, n_lstm_input]
        #print('cnn_sequential', len(cnn_sequential))    #.get_shape().as_list())
        #lstm_input = tf.contrib.keras.preprocessing.sequence.pad_sequences(cnn_sequential, maxlen=self.max_step, padding='post', truncating='post')     #输入不能为tensor？
        lstm_input = cnn_sequential
        rnn_output, fc_output, states = self.lstm('lstm1', lstm_input, n_layers=self.n_layers,
                                                  n_neurons=self.n_neurons, n_outputs=self.n_outputs,
                                                  sequence_length=self.n_steps)
        self.predict_label = fc_output
        return fc_output        # fc_output_size=(batch, max_step, n_outputs)  , first n_steps is no null

    def CNN_1D(self, scope_name, x, is_training, reuse=True):       #[batch, w, h, c]
        with tf.variable_scope(scope_name):
            conv_1 = self.conv('conv_1', x, ksize=7, filters_out=32, stride=2, is_training=is_training, batch_norm=True, reuse=reuse)
            norm_1 = tf.nn.local_response_normalization(conv_1)
            pool_1 = self.max_pool(norm_1, ksize=3, stride=2)

            conv_2 = self.conv('conv_2', pool_1, ksize=3, filters_out=64, stride=1, is_training=is_training, batch_norm=True, reuse=reuse)
            norm_2 = tf.nn.local_response_normalization(conv_2)
            pool_2 = self.max_pool(norm_2, ksize=3, stride=2)

            conv_3 = self.conv('conv_3', pool_2, ksize=3, filters_out=128, stride=1, is_training=is_training, batch_norm=True, reuse=reuse)
            pool_3 = self.max_pool(conv_3, ksize=3, stride=2)

            conv_4 = self.conv('conv_4', pool_3, ksize=3, filters_out=128, stride=1, is_training=is_training, batch_norm=True, reuse=reuse)
            pool_4 = self.max_pool(conv_4, ksize=3, stride=2)

            # with tf.variable_scope('scale_6'):  # feature size flap to 2 dim
            ave_pool_1 = tf.reduce_mean(pool_4, reduction_indices=[1, 2], name="avg_pool")  # feature size at batch*1*1*512
            n_input = ave_pool_1.get_shape().as_list()[-1]
            # print('n_input',n_input)
            output = tf.reshape(ave_pool_1, [self.batch_size, n_input])
            print('conv_1:', conv_1.get_shape().as_list())
            print('pool_1:', pool_1.get_shape().as_list())
            print('conv_2:', conv_2.get_shape().as_list())
            print('pool_2:', pool_2.get_shape().as_list())
            print('conv_3:', conv_3.get_shape().as_list())
            print('pool_3:', pool_3.get_shape().as_list())
            print('conv_4:', conv_4.get_shape().as_list())
            print('pool_4:', pool_4.get_shape().as_list())
            return output

    def CNN_2D(self):
        return

    def multi_CNN_output_1D(self, scope_name, x, is_training):     #x_size = [batch, n_step, w, h, c], sequential
        with tf.variable_scope(scope_name):
            # x_shape define in placeholder as None, thus we cannot use x.get_shape().as_list()[1] as n_steps
            n_steps =x.get_shape().as_list()[1]
            for i in range(n_steps):
                x_0 = x[:, i, :, :, :]
                if i == 0:
                    y_0 = self.CNN_1D('CNN_1', x_0, is_training, reuse=False)  # [batch, n_input]
                    y_0 = tf.expand_dims(y_0, 1)
                    y = y_0
                else:
                    y_0 = self.CNN_1D('CNN_1', x_0, is_training, reuse=True)  # [batch, n_input]
                    y_0 = tf.expand_dims(y_0, 1)
                    y = tf.concat([y,y_0], 1)

        return y        #y_size = [batch, n_step, n_feature, c]


    def multi_CNNoutput_2D(self, scope_name, x, is_training):     #x_size = [batch, n_step, w, h, c], sequential
        with tf.variable_scope(scope_name):
            # x_shape define in placeholder as None, thus we cannot use x.get_shape().as_list()[1] as n_steps
            n_steps =x.label.get_shape().as_list()[1]
            for i in range(n_steps):
                x_0 = x[:, i, :, :, :]
                if i == 0:
                    y_0 = self.CNN_2D('CNN_1', x_0, is_training, reuse=False)  # [batch, n_input]
                    y_0 = tf.expand_dims(y_0, 1)
                    y = y_0
                else:
                    y_0 = self.CNN_2D('CNN_1', x_0, is_training, reuse=True)  # [batch, n_input]
                    y_0 = tf.expand_dims(y_0, 1)
                    y = tf.concat([y,y_0], 1)

        return y

    def _loss(self, predict_label, labGT):     #label_predict_op=(batch, n_steps, n_outputs)  label=[batch, n_steps]
        weight_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=None)
        loss_weight = tf.add_n(weight_loss)

        # nax_step = predict_label.get_shape().as_list()[1]
        # for i in range(nax_step):
        #     predict_label_0 = predict_label[:, i, :]
        #     gt_label_0 = gt_label[:, i]
        #     ce_0 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict_label_0,
        #                                                           labels=gt_label_0)  # labels is in form of 1 dim,[batch_size], logistics is in 2 dims, [batch_size, output_num]
        #     ce_0 = tf.expand_dims(ce_0, 1)
		#
        #     prob_0 = tf.nn.softmax(predict_label_0)   #prob_size = [batch, n_outpts]
        #     prob_0 = tf.expand_dims(prob_0, 1)
        #     if i ==0:
        #         ce = ce_0
        #         prob = prob_0
        #     else:
        #         ce = tf.concat([ce, ce_0], 1)   #ce_shape = [batch, n_step]
        #         prob = tf.concat([prob, prob_0], 1)     #prob_size = [batch, n_step, n_outpts]
		#
        # #loss_direction =xxx
		#
        # loss_label = tf.reduce_sum(ce) / self.batch_size
        batch_size = labGT.get_shape().as_list()[0]
        n_step = labGT.get_shape().as_list()[1]
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict_label, labels=labGT)
        """labels = [batch_size, n_steps], logistics = [batch_size, n_steps, output_num]
        ce = [batch_size, n_steps]
        """
        print("predict_label", predict_label)
        print("labGT", labGT)
        print("ce", ce.get_shape().as_list())
        loss_label = tf.reduce_sum(ce) / (batch_size * n_step)

        #loss_p

        loss = loss_weight + loss_label #+ loss_direction
        self.loss = loss
        tf.summary.scalar('loss_train_each_batch_1', loss)

        return loss, loss_label, loss_weight



    def _loss_val(self, predict_label, label):
        batch_size = label.get_shape().as_list()[0]
        weight_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=None)
        loss_weight = tf.add_n(weight_loss)

        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict_label,
                                                            labels=label)  # labels is in form of 1 dim,[batch_size], logistics is in 2 dims, [batch_size, output_num]
        loss_label = tf.reduce_sum(ce)

        loss_label = loss_label / batch_size
        loss_weight = loss_weight

        loss_val = loss_weight + loss_label
        #tf.summary.scalar('loss_val', loss_val)     #recrease
        #print("loss_1", loss)
        return loss_val

    def _train(self):

        opt = tf.train.AdamOptimizer(self.init_lr, beta1=0.9, beta2=0.999, epsilon= 1e-08)

        gradients = opt.compute_gradients(self.loss)    #all variables are trainable
        apply_gradient_op = opt.apply_gradients(gradients)      #, global_step=self.global_step
        self.train = apply_gradient_op

        return apply_gradient_op

    def _top_k_accuracy(self, predict_label, labels, k=1):
        print('inacc',self.predict_label)
        batch_size = labels.get_shape().as_list()[0]
        n_steps = labels.get_shape().as_list()[1]
        acc_all = 0
        for i in range(n_steps):
            right_1 = tf.to_float(tf.nn.in_top_k(predictions=predict_label[:, i, :], targets=labels[:, i], k=k))
            num_correct_per_batch = tf.reduce_sum(right_1)
            acc = num_correct_per_batch / batch_size
            acc_all = acc_all + acc
        tf.summary.scalar('acc', acc_all/n_steps)

        return acc_all/n_steps

    # def print_size(self, x):
    #     print('conv_1:', x.get_shape().as_list())