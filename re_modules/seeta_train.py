import os
import time
import tensorflow as tf
from tensorflow.contrib import learn
from prepare_data.read_write_image import Dataset
from seeta_dataset import DataCenter
from net.resnet import get_resnet
import seetaas_helper as helper

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

NUM_LABEL = 64


class TrainModel:
    def __init__(self, cfg, dataset_path=None, dataset_name=None):
        self.mode = learn.ModeKeys.TRAIN
        self.start_iterm = 0
        self.gpus = [0]
        self.height = 64
        self.width = 256
        self.num_input_threads = 4
        self.cfg = cfg
        self.opt = tf.train.MomentumOptimizer(learning_rate=self.cfg.TRAIN.learning_rate,
                                              momentum=self.cfg.TRAIN.momentum)
        self.dataset = Dataset(batch_size=self.cfg.TRAIN.batch_size)
        self.seeta_dataset = DataCenter(dataset_path).load_dataset(dataset_name)
        self.example_size = self.seeta_dataset.record_count()
        self.max_iterm = int(self.example_size / self.cfg.TRAIN.batch_size * self.cfg.TRAIN.epoch)

    def train(self):
        global_step = tf.contrib.framework.get_or_create_global_step()
        train_op, loss, lr = self.get_net(global_step)
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.saver = tf.train.Saver()
        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        sess_config.gpu_options.allow_growth = True

        with tf.Session(config=sess_config) as sess:
            sess.run(init_op)
            display_freq = 20
            if cfg.TRAIN.restore:
                try:
                    ckpt = tf.train.get_checkpoint_state(self.cfg.TRAIN.output)
                    print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
                    self.saver.restore(sess, ckpt.model_checkpoint_path)
                    stem = os.path.splitext(os.path.basename(ckpt.model_checkpoint_path))[0]
                    start_iterm = int(stem.split('_')[-1])
                    sess.run(global_step.assign(start_iterm))
                    print('done')
                except:
                    raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

            generator = self.dataset._read_image(self.seeta_dataset)
            now = time.time()
            for step in range(self.start_iterm, self.max_iterm):
                data = next(generator)
                feed_dict = {self.input_images: data[0],
                             self.input_labels: data[1]}
                _, step_loss, lr_val, _ = sess.run([train_op, loss, lr, global_step], feed_dict=feed_dict)
                if step % display_freq == 0:
                    speed = display_freq * self.cfg.TRAIN.batch_size / (time.time() - now)
                    now = time.time()
                    print('[%s/%s] loss = %.4f speed = %.3f lr = %f'
                                % (step, self.max_iterm, step_loss, speed, lr_val))
            self.saver.save(sess, os.path.join(self.cfg.TRAIN.output, 'model.ckpt'), global_step=global_step)

    def get_net(self, global_step):
        self.input_images = tf.placeholder(tf.float32, shape=[self.cfg.TRAIN.batch_size, self.height, self.width, 3], name='input_images')
        self.input_labels = tf.placeholder(tf.int32, shape=(self.cfg.TRAIN.batch_size, ), name='input_labels')
        cls_score = get_resnet(self.input_images, is_training=True,
                               num_class=self.cfg.TRAIN.num_classes, batch_size=self.cfg.TRAIN.batch_size)
        label = tf.reshape(self.input_labels, [-1])
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))
        l2_losses = []
        for var in tf.trainable_variables():
            l2_losses.append(tf.nn.l2_loss(var))
        loss = tf.multiply(self.cfg.TRAIN.wd, tf.add_n(l2_losses)) + loss
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            learning_rate = tf.train.polynomial_decay(
                self.cfg.TRAIN.learning_rate,
                tf.train.get_global_step(),
                self.max_iterm,
                0.00001,
                power=0.6,
                name='learning_rate')
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=learning_rate,
                momentum=self.cfg.TRAIN.momentum
            )
            # global_step = tf.Variable(0, trainable=False)
            tvars = tf.trainable_variables()
            grads, norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), 10.0)
            train_op = optimizer.apply_gradients(list(zip(grads, tvars)), global_step=global_step)
        return train_op, loss, learning_rate


if __name__ == '__main__':
    cfg = helper.get_parameter(yaml_file=None)
    trainer = TrainModel(cfg, dataset_path=cfg.TRAIN.data_dir, dataset_name=None)
    trainer.train()
