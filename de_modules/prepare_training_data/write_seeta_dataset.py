import glob
import numpy as np
import scipy.sparse
import random
import cv2
import math
import os
import seeta_dataset as sd
import xml.etree.ElementTree as ET
from lib.fast_rcnn.config import cfg


class Pascal_voc(object):
    def __init__(self):
        self._classes = ('__background__',  # always index 0
                         'text')
        self.num_classes = len(self._classes)
        self._class_to_ind = dict(list(zip(self._classes, list(range(self.num_classes)))))

    def read_box(self, xml):
        tree = ET.parse(xml)
        objs = tree.findall('object')
        ret = []
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            cls = obj.find('name').text
            diffc = obj.find('difficult')
            difficult = 0 if diffc == None else int(diffc.text)
            ret.append({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
                        "name": cls, "difficult": difficult})
        return ret

    def _write(self, dataset_root, out_root, out_name):
        all_xml_path = glob.glob("{}/Annotations/*.xml".format(dataset_root))
        rt = sd.new_record_type({
            "content": sd.BasicType.ByteArray,
            "path": sd.BasicType.String,
            "object": [{
                "xmin": sd.BasicType.Int,
                "ymin": sd.BasicType.Int,
                "xmax": sd.BasicType.Int,
                "ymax": sd.BasicType.Int,
                "difficult": sd.BasicType.Int,
                "name": sd.BasicType.String,
            }]
        })
        print('>>>>', rt._rec)
        with sd.DataCenter(out_root).create_dataset(rt, out_name) as dataset:
            for i, xml_path in enumerate(all_xml_path):
                image_path = xml_path.replace("Annotations", "JPEGImages").replace(".xml", ".jpg")
                if not os.path.exists(image_path):
                    image_path = xml_path.replace("Annotations", "JPEGImages").replace(".xml", ".png")
                with open(image_path, 'rb') as fi:
                    content = fi.read()
                obj = self.read_box(xml_path)
                dataset.write({
                    "content": content,
                    "path": image_path,
                    "object": obj
                })
                if i % 100 == 0:
                    print("write #{}".format(i))

    def im_list_to_blob(self, ims):
        """Convert a list of images into a network input.

        Assumes images are already prepared (means subtracted, BGR order, ...).
        """
        max_shape = np.array([im.shape for im in ims]).max(axis=0)
        num_images = len(ims)
        blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                        dtype=np.float32)
        for i in range(num_images):
            im = ims[i]
            blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

        return blob

    def prep_im_for_blob(self, im, pixel_means, target_size, max_size):
        """Mean subtract and scale an image for use in a blob."""
        im = im.astype(np.float32, copy=False)
        im -= pixel_means
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        if cfg.TRAIN.RANDOM_DOWNSAMPLE:
            r = 0.6 + np.random.rand() * 0.4
            im_scale *= r
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)

        return im, im_scale

    def flipped_images(self, image, _dict):
        image = image[:, ::-1, :]
        height, width = image.shape[:2]
        boxes = _dict['boxes'].copy()
        oldx1 = boxes[:, 0].copy()
        oldx2 = boxes[:, 2].copy()
        boxes[:, 0] = width - oldx2
        boxes[:, 2] = width - oldx1
        for b in range(len(boxes)):
            if boxes[b][2] < boxes[b][0]:
                boxes[b][0] = 0
        assert (boxes[:, 2] > boxes[:, 0]).all()
        entry = {'image': image,
                 'boxes': boxes,
                 'gt_overlaps': _dict['gt_overlaps'],
                 'gt_classes': _dict['gt_classes'],
                 }
        if 'gt_ishard' in _dict and 'dontcare_areas' in _dict:
            entry['gt_ishard'] = _dict['gt_ishard'].copy()
            dontcare_areas = _dict['dontcare_areas'].copy()
            oldx1 = dontcare_areas[:, 0].copy()
            oldx2 = dontcare_areas[:, 2].copy()
            dontcare_areas[:, 0] = width - oldx2
            dontcare_areas[:, 2] = width - oldx1
            entry['dontcare_areas'] = dontcare_areas
        return entry

    def _seg_bbox(self, objs, image_size):
        data_list = []
        for obj in objs:
            xmin = int(float(obj['xmin']))
            ymin = int(float(obj['ymin']))
            xmax = int(float(obj['xmax']))
            ymax = int(float(obj['ymax']))
            assert xmin < xmax, ymin < ymax
            if xmin < 0:
                xmin = 0
            if xmax > image_size[1] - 1:
                xmax = image_size[1] - 1
            if ymin < 0:
                ymin = 0
            if ymax > image_size[0] - 1:
                ymax = image_size[0] - 1

            step = 24.0
            x_left = []
            x_right = []
            x_left.append(xmin)
            x_left_start = int(math.ceil(xmin / step) * step)
            if x_left_start == xmin:
                x_left_start = xmin + step
            for i in np.arange(x_left_start, xmax, step):
                x_left.append(i)
            x_left = np.array(x_left)

            x_right.append(x_left_start - 1)
            for i in range(1, len(x_left) - 1):
                x_right.append(x_left[i] + step - 1)
            x_right.append(xmax)
            x_right = np.array(x_right)

            idx = np.where(x_left == x_right)
            x_left = np.delete(x_left, idx, axis=0)
            x_right = np.delete(x_right, idx, axis=0)
            for i in range(len(x_left)):
                data_list.append({'xmin': x_left[i], 'ymin': ymin, 'xmax': x_right[i], 'ymax': ymax,
                                  'name': self._classes[1], 'difficult': obj['difficult']})
        return data_list

    def get_training_bbox(self, data):
        image = data['content']
        image = cv2.imdecode(np.fromstring(image, np.uint8), cv2.IMREAD_COLOR)
        image_path = data['path']
        objs = data['object']
        objs = self._seg_bbox(objs, image.shape[:2])
        num_objs = len(objs)
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ishards = np.zeros((num_objs), dtype=np.int32)
        dontcare_areas = np.zeros([0, 4], dtype=float)
        for ix, obj in enumerate(objs):
            x1 = obj['xmin']
            y1 = obj['ymin']
            x2 = obj['xmax']
            y2 = obj['ymax']
            cls = self._class_to_ind[obj['name'].lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
        overlaps = scipy.sparse.csr_matrix(overlaps)
        _dict = {'image': image,
                 'boxes': boxes,
                 'gt_classes': gt_classes,
                 'gt_ishard': ishards,
                 'gt_overlaps': overlaps,
                 'seg_areas': seg_areas,
                 'dontcare_areas': dontcare_areas
                 }
        if random.random() > 0.5:
            entry = self.flipped_images(image, _dict)
        else:
            entry = _dict
        target_size = cfg.TRAIN.SCALES[0]
        image, scale = self.prep_im_for_blob(entry['image'], cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE)
        image = self.im_list_to_blob([image])
        gt_inds = np.where(entry['gt_classes'] != 0)[0]
        gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
        gt_boxes[:, 0:4] = entry['boxes'][gt_inds, :] * scale
        gt_boxes[:, 4] = entry['gt_classes'][gt_inds]
        blobs = {'data': image}
        blobs['gt_boxes'] = gt_boxes
        blobs['gt_ishard'] = entry['gt_ishard'][gt_inds] \
            if 'gt_ishard' in entry else np.zeros(gt_inds.size, dtype=int)
        # blobs['gt_ishard'] = roidb[0]['gt_ishard'][gt_inds]
        blobs['dontcare_areas'] = entry['dontcare_areas'] * scale \
            if 'dontcare_areas' in entry else np.zeros([0, 4], dtype=float)
        blobs['im_info'] = np.array(
            [[image.shape[1], image.shape[2], scale]],
            dtype=np.float32)
        blobs['im_name'] = os.path.basename(image_path)

        return blobs


if __name__ == '__main__':
    root = "/tmp/image"
    poc = Pascal_voc()
    poc._write(dataset_root=root, out_root='/tmp/dataset', out_name='voc2007')
