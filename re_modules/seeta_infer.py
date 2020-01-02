import os
import copy
import cv2
import json
from math import sqrt
import numpy as np
import tensorflow as tf
from net.resnet import get_resnet
import seetaas_helper as helper
import seeta_dataset as sd
from seeta_dataset import DataCenter


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

IMG_HEIGHT = 64
IMG_WIDTH = 256
NUM_LABEL = 64
# _dict = {0: "am", 1: 'cw', 2: '2fsk', 3: '8fsk', 4: '8psk'}
_dict = {0: "2fsk", 1: '8fsk', 2: '8psk', 3: 'am', 4: 'cw'}


def new_dataset(origin_dataset, output_path):
    record_type = copy.deepcopy(origin_dataset._record_type)
    obj_type = sd.new_record_type(
        [{
            "xmin": sd.BasicType.Int,
            "ymin": sd.BasicType.Int,
            "xmax": sd.BasicType.Int,
            "ymax": sd.BasicType.Int,
            "score": sd.BasicType.Float,
            "difficult": sd.BasicType.Int,
            "name": sd.BasicType.String,
        }])

    cls_type = sd.new_record_type(
        {
            "name": sd.BasicType.String,
        })

    if 'object' not in record_type._keys and 'name' not in record_type._keys:
        record_type._properties["name"] = cls_type
    else:
        record_type._properties["object"] = obj_type
    return sd.DataCenter(output_path).create_dataset(record_type, "root")


def standardization(image):
    h, w = image.shape[:2]
    mean = np.mean(image)
    dev = np.std(image)
    num_elements = h * w * 3
    adjusted_stddev = max(dev, 1.0 / sqrt(num_elements))
    return (image - mean) / adjusted_stddev


class Infer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self._dict = _dict

    @staticmethod
    def get_image_batch(images, height, width):
        images_list = []
        for image in images:
            im = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
            im.astype(np.float32)
            im = standardization(im)
            images_list.append(im[None, ...])
        return images_list

    @staticmethod
    def get_rec_image(image, bboxes=None):
        images_list = []
        if bboxes:
            if len(bboxes) < 0:
                return []
            for bbox in bboxes:
                sub_image = image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :]
                im = cv2.resize(sub_image, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
                im.astype(np.float32)
                im = standardization(im)
                images_list.append(im[None, ...])
            return np.concatenate(images_list, 0)
        else:
            im = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
            im.astype(np.float32)
            im = standardization(im)
            return np.expand_dims(im, axis=0)

    @staticmethod
    def get_bboxes(objs):
        bboxes = []
        for obj in objs:
            bboxes.append([obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']])
        return bboxes

    def _read_infer_image(self, record_data, batch_size):
        input_image = []
        input_label = []
        for i, record in enumerate(record_data.random_read()):
            try:
                im = record["content"]
                label = self._dict[record['label']]
                image = cv2.imdecode(np.fromstring(im, np.uint8), cv2.IMREAD_COLOR)
                image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
                image = standardization(image)
                input_image.append(np.expand_dims(image.astype(np.float32), 0))
                input_label.append(label)
                if len(input_label) == batch_size:
                    yield (np.concatenate(input_image, 0), np.array(input_label, dtype=np.int32))
                    input_image = []
                    input_label = []
            except Exception as e:
                import traceback
                traceback.print_exc()
                continue
        return (np.concatenate(input_image, 0), np.array(input_label, dtype=np.int32))

    def _get_checkpoint(self):
        """Get the checkpoint path from the given model output directory"""
        ckpt = tf.train.get_checkpoint_state(self.cfg.TEST.checkpoints_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = ckpt.model_checkpoint_path
        else:
            raise RuntimeError('No checkpoint file found')
        return ckpt_path

    def infer(self):
        with tf.Graph().as_default():
            origin_image = tf.placeholder(tf.float32, shape=[None, 64, 256, 3])
            cls_score = get_resnet(origin_image, is_training=False, batch_size=tf.shape(origin_image)[0],
                                   num_class=self.cfg.TRAIN.num_classes)
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
                seeta_dataset = DataCenter(self.cfg.TEST.data_dir).load_dataset(None)
                total_num = seeta_dataset.record_count()
                output_dir = self.cfg.TEST.output
                seeta_dataset_dir = os.path.join(output_dir, self.cfg.TEST.output_name)
                if not os.path.exists(seeta_dataset_dir):
                    os.makedirs(seeta_dataset_dir)
                output_data = new_dataset(seeta_dataset, seeta_dataset_dir)
                if 'object' in output_data._record_type._keys:
                    for i, record in enumerate(seeta_dataset.read()):
                        print('========>> Progress: %s / %s' % (i, total_num))
                        image = record['content']
                        objs = copy.deepcopy(record['object'])
                        if len(objs) == 0:
                            record['object'] = []
                        else:
                            bboxes = self.get_bboxes(objs)
                            image = cv2.imdecode(np.fromstring(image, np.uint8), cv2.IMREAD_COLOR)
                            batch_images = self.get_rec_image(image, bboxes)
                            preds = sess.run([cls_pred], feed_dict={origin_image: batch_images})
                            labels = [self._dict[pred] for pred in preds[0]]
                            _k = objs[0].keys()
                            for i, label in enumerate(labels):
                                record['object'][i]['name'] = label
                                if 'score' not in _k:
                                    record['object'][i]['score'] = 1.0
                        output_data.write(record)
                else:
                    for i, record in enumerate(seeta_dataset.read()):
                        print('========>> Progress: %s / %s' % (i, total_num))
                        image = record['content']
                        image = cv2.imdecode(np.fromstring(image, np.uint8), cv2.IMREAD_COLOR)
                        image = self.get_rec_image(image)
                        output = sess.run(cls_pred, feed_dict={origin_image: image})
                        label = self._dict[output[0]]
                        record['name'] = label
                        output_data.write(record)

                seeta_dataset.close()
                output_data.close()
                json_dir = "{}/summary/".format(output_dir)
                if not os.path.exists(json_dir):
                    os.makedirs(json_dir)
                with open('%s/outputs.json' % json_dir, "w") as fo:
                    json.dump({
                        "output": [{
                            "name": "default",
                            "uuid": "reco_dataset"
                        }],
                    }, fo)


if __name__ == '__main__':
    cfg = helper.get_parameter(yaml_file=None)
    rec = Infer(cfg)
    rec.infer()
