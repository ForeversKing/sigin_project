import os
import cv2
import sys
from math import sqrt
import numpy as np
from seeta_dataset import new_record_type, DataCenter, BasicType
from prepare_data.data_util import GeneratorEnqueuer

pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])


def standardization(image):
    h, w = image.shape[:2]
    mean = np.mean(image)
    dev = np.std(image)
    num_elements = h * w * 3
    adjusted_stddev = max(dev, 1.0 / sqrt(num_elements))
    return (image - mean) / adjusted_stddev


class Dataset(object):
    def __init__(self, batch_size=32, out_dir=None, out_name=None, label_file=None, image_root=None):
        self._dict = {'am': 0, 'cw': 1, '2fsk': 2, '8fsk': 3, '8psk': 4}
        self.out_dir = out_dir
        self.out_name = out_name
        self.label_file = label_file
        self.image_root = image_root
        self.batch_size = batch_size
        self.IMG_HEIGHT = 64
        self.IMG_WIDTH = 256

    def _read_anno_file(self):
        with open(self.label_file) as fi:
            return fi.readlines()

    def _write_image(self):
        data_center = DataCenter(self.out_dir)
        rt = new_record_type({
            "path": BasicType.String,
            "name": BasicType.String,
            "content": BasicType.ByteArray,
        })
        with data_center.create_dataset(rt, self.out_name) as dataset:
            lines = self._read_anno_file()
            for line in lines:
                image_name, label = line.strip().split('\t')
                image_path = os.path.join(self.image_root, image_name)
                if not os.path.exists(image_path):
                    continue
                if os.stat(image_path).st_size == 0:
                    print('SKIPPING', image_path)
                    continue
                print("========== write image ===========")
                print("path: ", image_path)
                print("contentType", os.path.splitext(image_path)[1])
                if sys.version_info.major < 3:
                    with open(image_path, 'r') as fi:
                        content = fi.read()
                else:
                    with open(image_path, 'rb') as fi:
                        content = fi.read()
                dataset.write({
                    "path": image_path,
                    "name": label,
                    "content": content
                })

    def _batch_image(self):
        data_center = DataCenter(self.out_dir)
        dataset = data_center.load_dataset(self.out_name)
        input_image = []
        input_label = []
        for i, data in enumerate(dataset.random_read()):
            try:
                im = data["content"]
                # label = self._dict[data['name']]
                label = data['id']
                image = cv2.imdecode(np.fromstring(im, np.uint8), cv2.IMREAD_COLOR)
                image = cv2.resize(image, (self.IMG_WIDTH, self.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
                image = standardization(image)
                input_image.append(np.expand_dims(image.astype(np.float32), 0))
                input_label.append(label)
                if len(input_label) == self.batch_size:
                    yield (np.concatenate(input_image, 0), np.array(input_label, dtype=np.int32))
                    input_image = []
                    input_label = []
            except Exception as e:
                import traceback
                traceback.print_exc()
                continue

    def _read_image(self, record_data):
        input_image = []
        input_label = []
        while True:
            for i, record in enumerate(record_data.random_read()):
                try:
                    im = record["content"]
                    # label = self._dict[record['name']]
                    label = record['id']
                    image = cv2.imdecode(np.fromstring(im, np.uint8), cv2.IMREAD_COLOR)

                    # image = image.astype(np.float32, copy=False)
                    # image -= pixel_means

                    image = cv2.resize(image, (self.IMG_WIDTH, self.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
                    image = standardization(image)
                    input_image.append(np.expand_dims(image.astype(np.float32), 0))
                    input_label.append(label)
                    if len(input_label) == self.batch_size:
                        yield (np.concatenate(input_image, 0), np.array(input_label, dtype=np.int32))
                        input_image = []
                        input_label = []
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    continue

    def get_batch(self, num_workers):
        try:
            enqueuer = GeneratorEnqueuer(self._read_image(), use_multiprocessing=True)
            print('Generator use 10 batches for buffering, this may take a while, you can tune this yourself.')
            enqueuer.start(max_queue_size=10, workers=num_workers)
            generator_output = None
            while True:
                while enqueuer.is_running():
                    if not enqueuer.queue.empty():
                        generator_output = enqueuer.queue.get()
                        break
                    else:
                        time.sleep(1)
                yield generator_output
                generator_output = None
        finally:
            if enqueuer is not None:
                enqueuer.stop()


if __name__ == '__main__':
    label_file = '/home/yulongwu/d/data/sign_data/recognize_data/1/label.txt'
    image_root = '/home/yulongwu/d/data/sign_data/recognize_data/1/image'
    out_dir = '../dataset'
    out_name = 'rec_data'
    dataset = Dataset(out_dir=out_dir, out_name=out_name, image_root=image_root, label_file=label_file)
    dataset._write_image()
