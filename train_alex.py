import tensorflow as tf
import random
import CNN_ALEX as Network
import time
import numpy as np
import math
import os

flags = tf.app.flags

flags.DEFINE_integer("epoch", 2500, "Epoch to train [25]")
flags.DEFINE_integer("batch_size", 1, "The size of batch images")
#flags.DEFINE_integer("max_step", 15, "The size of lstm time steps")
flags.DEFINE_integer("n_steps", 5, "The size of lstm time steps")
flags.DEFINE_integer("n_layers", 2, "The size of lstm layer width")
flags.DEFINE_integer("n_neurons", 256, "The size of lstm hidden cell")
flags.DEFINE_integer("n_outputs", 2, "The size of output classification")

flags.DEFINE_string("model_name", "4.29", "name the save model")
flags.DEFINE_string("model_save_dir", "../model/", "address of the save model")
flags.DEFINE_string("summary_save_dir", "../summary/", "address of the training summary")
flags.DEFINE_string("tfrecord_train", "G:\sequential/clstm_liliu/train_2164_step_5.tfrecords", "address of the tfrecord file for train")
flags.DEFINE_string("tfrecord_test", "G:\sequential/clstm_liliu/test_532_step_5.tfrecords", "address of the tfrecord file for test")

# 现在 input image 为[batch, 15,w,h,c], label仍为[batch, n_step]

flags.DEFINE_float("init_lr", 10 ** (-4), "Initial learning rate")
flags.DEFINE_integer("randon_seed", 419, "The number of random seed")
flags.DEFINE_string("GPU_No", "0", "The number of GPU")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")

FLAGS = flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU_No
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

random.seed(a=FLAGS.randon_seed)
tf.set_random_seed(FLAGS.randon_seed)
randseed = FLAGS.randon_seed


iter_save = 2000
iter_validation = 10
input_size = [224, 224]
output_size = [112, 112]
validation = True

batch_size = 4
epoch_num = 300

batch_size_val = 4
validation_num = 532
train_num = 2164

epoch_num_val = int((train_num / FLAGS.batch_size * FLAGS.epoch) / iter_validation) + 1

thre = 0.5
e = math.e

def main():


    """START"""
    net = Network.Net(
        batch_size=FLAGS.batch_size,
        n_steps=FLAGS.n_steps,
        n_layers=FLAGS.n_layers,
        n_neurons=FLAGS.n_neurons,
        n_outputs=FLAGS.n_outputs,
        init_lr=FLAGS.init_lr,
       # max_step=FLAGS.max_step
    )
    net.trainable = True
    #sequence_length = tf.placeholder(tf.int64, (FLAGS.batch_size))
    input = tf.placeholder(tf.float32, (FLAGS.batch_size, FLAGS.n_steps, input_size[0], input_size[1], 3))
    GT_label = tf.placeholder(tf.int64, (FLAGS.batch_size, FLAGS.n_steps))   #size = [batch, n_steps]

    label_predict_op = net.inference(input)  # label_predict_op=(batch, n_steps, n_outputs)
    loss_op, loss_label_op, loss_weight_op = net._loss(label_predict_op, GT_label)   #[batch,n_steps]

    loss_op_v = net._loss_val(label_predict_op, GT_label)
    acc_op = net._top_k_accuracy(label_predict_op, GT_label)

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_op = net._train()

    sess = tf.Session(config=config)
    saver = tf.train.Saver(max_to_keep=20)
    init = tf.global_variables_initializer()
    sess.run(init)

    summary_op = tf.summary.merge_all()
    summary_train_writer = tf.summary.FileWriter(FLAGS.summary_save_dir + FLAGS.model_name , sess.graph)

    def _parse_function(example_proto):
        keys_to_features = {
                            'shape': tf.FixedLenFeature([], tf.string),
                            'GTlabel': tf.FixedLenFeature([], tf.string),
                            'image': tf.FixedLenFeature([], tf.string),
                            'patient_name': tf.FixedLenFeature([], tf.string),
                            'n_step': tf.FixedLenFeature([], tf.int64),
                            'year': tf.FixedLenFeature([], tf.string),
                            }
        parsed_features = tf.parse_single_example(example_proto, keys_to_features, name='features')
        shape = tf.decode_raw(parsed_features['shape'], tf.int32)
        GTlabel = tf.decode_raw(parsed_features['GTlabel'], tf.int32)  # but GTlabel is int64?
        image = tf.decode_raw(parsed_features['image'], tf.uint8)
        patient_name = tf.decode_raw(parsed_features['patient_name'], tf.int8)
        n_step = parsed_features['n_step']
        year = tf.decode_raw(parsed_features['year'], tf.int32)
        image = tf.reshape(image, [-1, shape[0], shape[1], 3])
        image_224uint = tf.cast(image, dtype=tf.uint8)

        #return shape, GTlabel, patient_name, year, n_step  #parsed_features['name']  image_224uint,

        return shape, GTlabel, image_224uint, patient_name, n_step, year

    """train_dataset"""
    dataset_train = tf.data.TFRecordDataset(FLAGS.tfrecord_train)
    dataset_train = dataset_train.map(_parse_function)
    dataset_train = dataset_train.shuffle(buffer_size=1000).batch(FLAGS.batch_size).repeat(FLAGS.epoch)
    iterator = dataset_train.make_initializable_iterator()
    shape, GTlabel, image, patient_name, n_step, year = iterator.get_next()#image,

    sess.run(iterator.initializer)

    if validation:
        dataset_test = tf.data.TFRecordDataset(FLAGS.tfrecord_test)
        dataset_test = dataset_test.map(_parse_function)
        dataset_test = dataset_test.batch(FLAGS.batch_size).repeat(epoch_num_val)
        iterator_test = dataset_test.make_initializable_iterator()
        shape_v, GTlabel_v, image_v, patient_name_v, n_step_v, year_v = iterator_test.get_next()
        sess.run(iterator_test.initializer)

    iter = 0
    start_time = time.time()
    loss_list = np.array([])
    acc_val_thre_list = np.array([])
    countdata = 0

    while True:
        try:
            image1, GTlabel1, n_step1 = sess.run([image, GTlabel, n_step])
            #print(image1)
            #c1 = image1[:, 0:5, :, :, :]

            countdata = countdata + 1
            iter = iter + 1

            """train"""
            net.is_training = True
            assert net.is_training == True
            summary_train, loss_train, loss_label_train, loss_weight_train, _, acc, label_predict= sess.run(
                [summary_op, loss_op, loss_label_op, loss_weight_op, train_op, acc_op, label_predict_op],
                feed_dict={input: image1[:, 0:5, :, :, :], GT_label: GTlabel1[:, 1:6]})

            acc_thre = net._result(labGTs=GTlabel1[:, 1:6], predictions=label_predict)

            summary1 = tf.Summary(value=[
                tf.Summary.Value(tag="loss_train_each_batch_2", simple_value=loss_train),  # increase
                tf.Summary.Value(tag="acc_train_each_batch_1", simple_value=acc),       # from top_k_acc
                tf.Summary.Value(tag="acc_train_each_batch_2", simple_value=acc_thre),  # 训练集一个batch的准确率 from _result
            ])
            summary_train_writer.add_summary(summary1, iter)

            assert not np.isnan(loss_train)
            summary_train_writer.add_summary(summary_train, iter)
            print('Training: iteration=', iter, '  Loss=', loss_train, '  Acc=', acc_thre)

            """save"""
            if iter % iter_save == 0:  # 5000   #only save model
                saver.save(sess, FLAGS.model_save_dir + '/' + FLAGS.model_name + '/cnnlstm', global_step=iter)
                print('It took', time.time() - start_time, 'time')
                start_time = time.time()

            """validation"""
            if iter % iter_validation == 0 and validation:  # 100
                print('Validation waiting...')
                acc_val_thre_list=[]
                while True:
                    for i in range(int(validation_num / batch_size_val)):
                        net.is_training = False
                        assert net.is_training == False  # and net.Is_training == False

                        image2, GTlabel2, n_step_v2, patient_name_v2 = sess.run([image_v, GTlabel_v, n_step_v, patient_name_v])
                        predict_label = sess.run(label_predict_op, feed_dict={input: image2[:, 0:5, :, :, :]})
                        loss_val = sess.run(loss_op_v,
                            feed_dict={input: image2[:, 0:5, :, :, :], GT_label: GTlabel2[:, 1:6]})



                        assert not np.isnan(loss_val)
                        loss_list = np.append(loss_list, loss_val)

                        acc_val_thre = net._result(labGTs=GTlabel2[:, 1:6], predictions=predict_label)      #存疑

                        acc_val_thre_list = np.append(acc_val_thre_list, acc_val_thre)

                        summary1 = tf.Summary(value=[
                            tf.Summary.Value(tag="loss_val_each_batch", simple_value=loss_val),  # increase
                            tf.Summary.Value(tag="acc_val_each_batch", simple_value=acc_val_thre),  # increase
                        ])
                        summary_train_writer.add_summary(summary1, iter)

                    acc_thre = np.mean(acc_val_thre_list)       #在整个测试集上的acc
                    loss_validation = np.mean(loss_list)
                    summary = tf.Summary(value=[
                        tf.Summary.Value(tag="loss_val_mean", simple_value=loss_validation),         #increase
                        tf.Summary.Value(tag="acc_val_mean", simple_value=acc_thre),     #整个测试集的准确率
                    ])
                    summary_train_writer.add_summary(summary, iter)

                    print('Testing: iteration=', iter, '  Loss=', loss_validation, '  Acc_thre=', acc_thre,)

                    """save_variable"""

                    loss_list = np.array([])
                    tp_list = np.array([])
                    tn_list = np.array([])
                    fp_list = np.array([])
                    fn_list = np.array([])
                    acc_val_thre_list = np.array([])

                    print('set to zero.')
                    break


        except tf.errors.OutOfRangeError:
            break


if __name__ == '__main__':
    main()


