import os
import glob
import random
import tensorflow as tf
import numpy as np
import cv2
import time
from net.resnet import get_resnet

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

IMG_HEIGHT = 64
IMG_WIDTH = 256
NUM_LABEL = 64
_dict = {'am': 0, 'cw': 1, '2fsk': 2, '8fsk': 3, '8psk': 4}


class Infer:
    def __init__(self):
        # self.model = '/home/yulongwu/d/ocr/text-detection-ctpn/output/recognize_model'
        self.model = '/tmp/model/'
        self.image_root = '/home/yulongwu/d/data/sign_data/recognize_data/0/test_data/image'
        self.label_file = '/home/yulongwu/d/data/sign_data/recognize_data/0/test_data/label.txt'
        self.num_classes = 5

    # @staticmethod
    # def get_image_batch(im_path):
    #     max_ratio = NUM_LABEL
    #     img = cv2.imread(im_path)
    #     shape = img.shape
    #     hight = shape[0]
    #     width = shape[1]
    #     ratio = (1.0 * width / hight)
    #     if ratio > max_ratio:
    #         ratio = max_ratio
    #     if ratio < 1:
    #         ratio = 1
    #     # img = misc.imresize(img, (IMG_HEIGHT, int(IMG_HEIGHT * ratio)), interp='bilinear')
    #     img = cv2.resize(img, (int(IMG_HEIGHT * ratio), IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
    #     width = int(IMG_HEIGHT * ratio)
    #     new_width = width + (IMG_WIDTH - width % IMG_WIDTH)
    #     if new_width - width > 0:
    #         new_img = np.zeros((IMG_HEIGHT, new_width, 3), dtype=np.uint8)
    #         for i in range(3):
    #             padding_value = int(np.mean(img[:][-1][i]))
    #             z = np.ones((IMG_HEIGHT, new_width - width), np.uint8) * padding_value
    #             new_img[:, :, i] = np.hstack((img[:, :, i], z))
    #     else:
    #         new_img = img
    #     num = int(new_width / IMG_WIDTH)
    #     images = []
    #     for j in range(num):
    #         images.append(new_img[:, IMG_WIDTH * j: IMG_WIDTH * (j + 1), :])
    #     return images

    @staticmethod
    def get_image_batch(im_path):
        img = cv2.imread(im_path)
        shape = img.shape
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
        return [img]

    def generate_test_image(self):
        elements = []
        with open(self.label_file) as fi:
            lines = fi.readlines()
            for line in lines:
                image_name, label = line.strip().split('\t')
                image_path = os.path.join(self.image_root, image_name)
                if os.stat(image_path).st_size == 0:
                    continue
                images = self.get_image_batch(image_path)
                for image in images:
                    elements.append((image, _dict[label]))
        random.shuffle(elements)
        return elements

    def _get_checkpoint(self):
        """Get the checkpoint path from the given model output directory"""
        ckpt = tf.train.get_checkpoint_state(self.model)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = ckpt.model_checkpoint_path
        else:
            raise RuntimeError('No checkpoint file found')
        return ckpt_path

    def infer(self):
        with tf.Graph().as_default():
            origin_image = tf.placeholder(tf.float32, shape=[64, 256, 3])
            image = tf.image.per_image_standardization(origin_image)
            image = tf.expand_dims(image, 0)
            cls_score = get_resnet(image, is_training=False, batch_size=1, num_class=self.num_classes)
            cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")
            session_config = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,
                gpu_options=gpu_options)
            saver = tf.train.Saver()
            with tf.Session(config=session_config) as sess:
                checkpoint = self._get_checkpoint()
                print('Restore from: ', checkpoint)
                saver.restore(sess, checkpoint)
                print('Restore over ~~~')
                total = 0
                count = 0
                elements = self.generate_test_image()
                for element in elements:
                    output = sess.run([cls_pred], feed_dict={origin_image: element[0]})
                    if int(element[1]) == int(output[0]):
                        count += 1
                    total += 1
            print('the total test image number is: %s, precision is : %s' % (total, count / total))


if __name__ == '__main__':
    rec = Infer()
    rec.infer()
