import os
import time
import logging
import tensorflow as tf
from tensorflow.contrib import learn
from re_modules.prepare_data import read_tfrecord
from re_modules.net.resnet import get_resnet
logging.basicConfig(filename='./log.txt',
                    filemode='a+',
                    format='%(asctime)s-%(name)s-%(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

NUM_LABEL = 64


class TrainModel:
    def __init__(self):
        self.mode = learn.ModeKeys.TRAIN
        self.output = '/home/yulongwu/d/ocr/text-detection-ctpn/output/recognize_model'
        self.tune_from = ''  # Path to pre-trained model checkpoint
        self.tune_scope = ''  # Variable scope for training
        self.batch_size = 32
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.wd = 0.00002
        self.epoch = 100
        self.example_size = 2000
        self.num_classes = 5
        self.max_num_steps = int(self.example_size / self.batch_size * self.epoch)
        self.decay_steps = self.max_num_steps
        self.gpus = [0]
        self.train_device = '/gpu:1'  # Device for preprocess/batching graph placement
        self.train_path = '/home/yulongwu/d/data/sign_data/recognize_data/1/tf_record'
        # self.train_path = '/home/yulongwu/d/ocr/data/arti_data/chinese_data/supple_data/tf_record'
        self.filename_pattern = 'tf-*.tfrecord'  # File pattern for input data
        self.num_input_threads = 4
        self.opt = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                              momentum=self.momentum)

    def train(self):
        with tf.Graph().as_default():
            global_step = tf.contrib.framework.get_or_create_global_step()
            train_op, lr = self.get_net()
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            sv = tf.train.Supervisor(
                logdir=self.output,
                init_op=init_op,
                # summary_op=summary_op,
                save_summaries_secs=3000,
                init_fn=self._get_init_pretrained(),
                save_model_secs=3000)

            sess_config = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)
            sess_config.gpu_options.allow_growth = True
            with sv.managed_session(config=sess_config) as sess:
                step = sess.run(global_step)
                logger.info('init step: %s', step)
                now = time.time()
                display_freq = 20
                while step < self.max_num_steps:
                    if sv.should_stop():
                        break
                    step_loss, step, lr_val = \
                        sess.run([train_op, global_step, lr])
                    if step % display_freq == 0:
                        speed = display_freq * self.batch_size / (time.time() - now)
                        now = time.time()
                        logger.info('[%s/%s] loss = %s speed = %s lr = %s'
                                    % (step, self.max_num_steps, step_loss, speed, lr_val))
                sv.saver.save(sess, os.path.join(self.output, 'model.ckpt'), global_step=global_step)

    def get_net(self):
        image, label = read_tfrecord.read_and_decode(
            self.train_path,
            batch_size=self.batch_size,
            num_threads=self.num_input_threads)
        with tf.device(self.train_device):
            cls_score = get_resnet(image, is_training=True, num_class=self.num_classes, batch_size=self.batch_size)
            label = tf.reshape(label, [-1])
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))
            l2_losses = []
            for var in tf.trainable_variables():
                l2_losses.append(tf.nn.l2_loss(var))
            logger.info('l2_losses:\n %s', l2_losses)
            loss = tf.multiply(self.wd, tf.add_n(l2_losses)) + loss
            # Update batch norm stats [http://stackoverflow.com/questions/43234667]
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                learning_rate = tf.train.polynomial_decay(
                    self.learning_rate,
                    tf.train.get_global_step(),
                    self.decay_steps,
                    0.00001,
                    power=0.6,
                    name='learning_rate')
                # optimizer = tf.train.AdamOptimizer(
                #     learning_rate=learning_rate)
                optimizer = tf.train.MomentumOptimizer(
                    learning_rate=learning_rate,
                    momentum=self.momentum
                )
                train_op = tf.contrib.layers.optimize_loss(
                    loss=loss,
                    global_step=tf.train.get_global_step(),
                    learning_rate=learning_rate,
                    optimizer=optimizer)
        return train_op, learning_rate

    def _get_init_pretrained(self):
        if self.tune_from:
            logger.info('Restore From: %s', self.tune_from)
            saver_reader = tf.train.Saver(
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
            ckpt_path = self.tune_from
            return lambda sess: saver_reader.restore(sess, ckpt_path)
        return None


if __name__ == '__main__':
    trainer = TrainModel()
    trainer.train()
