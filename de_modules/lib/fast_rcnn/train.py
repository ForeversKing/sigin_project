from __future__ import print_function
import os
import tensorflow as tf
from lib.fast_rcnn.config import cfg
from lib.utils.timer import Timer
from seeta_dataset import DataCenter
from prepare_training_data.write_seeta_dataset import Pascal_voc

_DEBUG = False


class SolverWrapper(object):
    def __init__(self, train_graph, network, output_dir,
                 pretrained_model=None, test_net=None, dataset_path=None, dataset_name=None):
        """Initialize the SolverWrapper."""
        self.net = network
        self.dataset = Pascal_voc()
        self.test_net = test_net
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model
        self.train_graph = train_graph
        self.seeta_dataset = DataCenter(dataset_path).load_dataset(dataset_name)
        self.max_epoch = cfg.TRAIN.EPOCH
        self.example_size = self.seeta_dataset.record_count()
        self.max_iter = self.example_size * self.max_epoch
        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    def snapshot(self, sess, iter):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
                    '_iter_{:d}'.format(iter + 1) + '.ckpt')
        filename = os.path.join(self.output_dir, filename)

        self.saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

    def train_model(self, sess):
        """Network training loop."""
        total_loss, model_loss, rpn_cross_entropy, rpn_loss_box = self.net.build_loss(ohem=cfg.TRAIN.OHEM)
        # scalar summary
        tf.summary.scalar('rpn_reg_loss', rpn_loss_box)
        tf.summary.scalar('rpn_cls_loss', rpn_cross_entropy)
        tf.summary.scalar('model_loss', model_loss)
        tf.summary.scalar('total_loss', total_loss)

        learning_rate = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, self.max_iter, decay_steps=4000,
                                                   decay_rate=0.90, staircase=True)
        lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
        if cfg.TRAIN.SOLVER == 'Adam':
            opt = tf.train.AdamOptimizer(learning_rate)
        elif cfg.TRAIN.SOLVER == 'RMS':
            opt = tf.train.RMSPropOptimizer(learning_rate)
        else:
            # lr = tf.Variable(0.0, trainable=False)
            momentum = cfg.TRAIN.MOMENTUM
            opt = tf.train.MomentumOptimizer(lr, momentum)

        global_step = tf.Variable(0, trainable=False)
        with_clip = True
        if with_clip:
            tvars = tf.trainable_variables()
            grads, norm = tf.clip_by_global_norm(tf.gradients(total_loss, tvars), 10.0)
            train_op = opt.apply_gradients(list(zip(grads, tvars)), global_step=global_step)
        else:
            train_op = opt.minimize(total_loss, global_step=global_step)

        # intialize variables
        sess.run(tf.global_variables_initializer())
        start_epoch = 0

        # load vgg16
        if self.pretrained_model is not None and not cfg.TRAIN.restore:
            try:
                print(('Loading pretrained model '
                       'weights from {:s}').format(self.pretrained_model))
                self.net.load(self.pretrained_model, sess, True)
            except:
                raise 'Check your pretrained model {:s}'.format(self.pretrained_model)

        # resuming a trainer
        if cfg.TRAIN.restore:
            try:
                ckpt = tf.train.get_checkpoint_state(self.output_dir)
                print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                stem = os.path.splitext(os.path.basename(ckpt.model_checkpoint_path))[0]
                start_epoch = int(stem.split('_')[-1])
                sess.run(global_step.assign(start_epoch))
                print('done')
            except:
                raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

        timer = Timer()
        for epoch in range(start_epoch, self.max_epoch):
            det = self.seeta_dataset
            for iter, data in enumerate(det.random_read()):
                timer.tic()
                if epoch != 0 and (epoch * self.example_size + iter) % cfg.TRAIN.STEPSIZE == 0:
                    sess.run(tf.assign(lr, lr.eval() * cfg.TRAIN.GAMMA))
                    print(lr)

                blobs = self.dataset.get_training_bbox(data)
                feed_dict = {
                    self.net.data: blobs['data'],
                    self.net.im_info: blobs['im_info'],
                    self.net.keep_prob: 0.5,
                    self.net.gt_boxes: blobs['gt_boxes'],
                    self.net.gt_ishard: blobs['gt_ishard'],
                    self.net.dontcare_areas: blobs['dontcare_areas']
                }

                res_fetches = []
                fetch_list = [total_loss, model_loss, rpn_cross_entropy, rpn_loss_box,
                              train_op] + res_fetches

                total_loss_val, model_loss_val, rpn_loss_cls_val, rpn_loss_box_val, _ = sess.run(fetches=fetch_list,
                                                                                                 feed_dict=feed_dict)
                _diff_time = timer.toc(average=False)

                if (epoch * self.example_size + iter) % (cfg.TRAIN.DISPLAY) == 0:
                    print(
                        'iter: %d / %d, total loss: %.4f, model loss: %.4f, rpn_loss_cls: %.4f, rpn_loss_box: %.4f, lr: %f' % \
                        (epoch * self.example_size + iter, self.max_iter, total_loss_val, model_loss_val,
                         rpn_loss_cls_val, rpn_loss_box_val, lr.eval()))
                    print('speed: {:.3f}s / iter'.format(_diff_time))
            if epoch:
                self.snapshot(sess, epoch)
