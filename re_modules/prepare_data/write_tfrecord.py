import tensorflow as tf
import math
import os
import logging
import cv2
import random
import numpy as np
import scipy.misc as misc


def _int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def make_example(image_data, labels, height, width):
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(tf.compat.as_bytes(image_data)),
        'image/labels': _int64_feature([labels]),
        'image/height': _int64_feature([height]),
        'image/width': _int64_feature([width]),
    }))
    return example


IMG_HEIGHT = 64
IMG_WIDTH = 256
NUM_LABEL = 64
_dict = {'am': 0, 'cw': 1, '2fsk': 2, '8fsk': 3, '8psk': 4}


class CreateRecord:
    def __init__(self, root, anno_file):
        self._root_path = root
        self._input = self._read_anno_file(anno_file)
        self.jpeg_data = tf.placeholder(dtype=tf.string)
        self.jpeg_decoder = tf.image.decode_jpeg(self.jpeg_data,
                                                 channels=1)

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
    #     num = int(new_width/IMG_WIDTH)
    #     images = []
    #     for j in range(num):
    #         images.append(new_img[:, IMG_WIDTH*j: IMG_WIDTH*(j + 1), :])
    #     return images

    @staticmethod
    def get_image_batch(im_path):
        img = cv2.imread(im_path)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
        return [img]

    @staticmethod
    def _read_anno_file(anno_file):
        with open(anno_file) as fi:
            return fi.readlines()

    def _write_record(self, _root, input_list, save_rec_file):
        writer = tf.python_io.TFRecordWriter(save_rec_file)
        elements = []
        for input in input_list:
            im_name, label = input.strip().split('\t')
            label = _dict[label]
            image_path = os.path.join(_root, im_name)
            if not os.path.exists(image_path):
                continue
            if os.stat(image_path).st_size == 0:
                print('SKIPPING', image_path)
                continue

            images = self.get_image_batch(image_path)
            for image in images:
                elements.append((image, label))
        random.shuffle(elements)
        try:
            for element in elements:
                # string = cv2.imencode('.jpeg', element[0])[1].tostring()
                string = element[0].tobytes()
                example = make_example(string, element[1], element[0].shape[0], element[0].shape[1])
                writer.write(example.SerializeToString())
        except:
            logging.exception('error')
            print('ERROR', image_path)
        writer.close()

    def create(self, output_prefix, start_shard=0, num_shards=2):
        num_digits = math.ceil(math.log10(num_shards - 1))
        shard_format = '%0' + '%d' % num_digits + 'd'  # Use appropriate # leading zeros
        images_per_shard = int(math.ceil(len(self._input) / float(num_shards)))
        for i in range(start_shard, num_shards):
            start = i * images_per_shard
            end = (i + 1) * images_per_shard
            rec_file = '%s-%s.tfrecord' % (output_prefix, (shard_format % i))
            if os.path.isfile(rec_file):  # Don't recreate data if restarting
                continue
            self._write_record(self._root_path, self._input[start: end], rec_file)
            print('%s of %s [%s: %s] %s' % (i, num_shards, start, end, rec_file))
        start = num_shards * images_per_shard
        rec_file = '%s-%s.tfrecord' % (output_prefix, (shard_format % num_shards))
        self._write_record(self._root_path, self._input[start:], rec_file)


if __name__ == '__main__':
    label_path = '/home/yulongwu/d/data/sign_data/recognize_data/1/label.txt'
    _root = '/home/yulongwu/d/data/sign_data/recognize_data/1/image'
    creator = CreateRecord(root=_root, anno_file=label_path)
    creator.create('/home/yulongwu/d/data/sign_data/recognize_data/1/tf_record/tf')
