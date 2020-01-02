from __future__ import print_function
import cv2
import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# sys.path.append(os.getcwd())
import json
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.fast_rcnn.test import test_ctpn
from lib.utils.timer import Timer
from seeta_dataset import DataCenter
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg
import seetaas_helper as helper

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import seeta_dataset as sd
import copy


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
    if 'object' not in record_type._keys:
        # record_type._keys.append("object")
        record_type._properties["object"] = obj_type
    output_data = sd.DataCenter(output_path).create_dataset(record_type, "root")
    return output_data


def calculateIoU(candidateBound, groundTruthBound):
    cx1 = candidateBound[0]
    cy1 = candidateBound[1]
    cx2 = candidateBound[2]
    cy2 = candidateBound[3]

    gx1 = groundTruthBound[0]
    gy1 = groundTruthBound[1]
    gx2 = groundTruthBound[2]
    gy2 = groundTruthBound[3]

    carea = (cx2 - cx1) * (cy2 - cy1)  # C的面积
    garea = (gx2 - gx1) * (gy2 - gy1)  # G的面积

    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = w * h  # C∩G的面积

    iou_candidate = area / carea
    iou_groundtruth = area / garea

    return iou_candidate, iou_groundtruth


def tmp_contrate(images):
    master_image = np.zeros((10 * 1024, 1024, 3), dtype=np.uint8)
    for i in range(len(images)):
        image = cv2.resize(images[i], (1024, 1024))
        master_image[i * 512:(i + 1) * 512, :, :] = image[:512, :, :]
    return master_image


def process_boxes(texts_boxes):
    texts_boxes = np.array(texts_boxes)
    tmp_texts_boxes = texts_boxes.copy()
    number_boxes = tmp_texts_boxes.shape[0]
    remove_ixs = []
    for i in range(number_boxes - 1):
        for j in range(i + 1, number_boxes):
            w_i = tmp_texts_boxes[i][2] - tmp_texts_boxes[i][0]
            w_j = tmp_texts_boxes[j][2] - tmp_texts_boxes[j][0]
            iou_i, iou_j = calculateIoU(tmp_texts_boxes[i], tmp_texts_boxes[j])
            if iou_i != iou_j:
                if iou_i <= 0.1 and iou_j <= 0.1:
                    continue
                if w_i < w_j:
                    remove_ixs.append(i)
                elif w_j < w_i:
                    remove_ixs.append(j)
                else:
                    if iou_i < iou_j:
                        remove_ixs.append(j)
                    elif iou_j < iou_i:
                        remove_ixs.append(i)
            elif iou_i == iou_j == 1.0:
                remove_ixs.append(i)

    remove_ixs = np.unique(np.array(remove_ixs))
    texts_boxes = np.delete(texts_boxes, remove_ixs, axis=0)
    return texts_boxes


def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f


def cv_vis(im, dets, save_path=None):
    colors = [(0, 255, 0), (255, 255, 0), (0, 255, 255)]
    for det in dets:
        color = random.choice(colors)
        im = cv2.rectangle(im, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), color, 2)
        score = det[4]
        im = cv2.putText(im, '{:.3f}'.format(score), (int((det[0] + det[2]) / 2), int((det[3] + det[1]) / 2)),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return im


def draw_boxes(img, boxes, scale):
    text_boxes = []
    for box in boxes:
        if np.linalg.norm(box[0] - box[2]) < 5 or np.linalg.norm(box[1] - box[7]) < 5:
            continue
        min_x = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
        min_y = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
        max_x = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
        max_y = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
        text_boxes.append((min_x, min_y, max_x, max_y, box[-1]))
    if len(text_boxes) > 0:
        text_boxes = process_boxes(text_boxes)
    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
    # im = cv_vis(img, text_boxes)
    return img, text_boxes


def ctpn(sess, net, img):
    timer = Timer()
    timer.tic()
    img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    scores, boxes = test_ctpn(sess, net, img)
    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    sort_index = np.argsort(boxes[:, -1])[::-1]
    boxes = boxes[sort_index]
    im, bboxes = draw_boxes(img, boxes, scale)
    timer.toc()
    print(('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0]))
    return im, bboxes


if __name__ == '__main__':
    yaml_cfg = helper.get_parameter(yaml_file=None)
    cfg_from_file(yaml_cfg)
    print(cfg)
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    net = get_network("VGGnet_test")
    print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
    saver = tf.train.Saver()
    try:
        ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

    seeta_dataset = DataCenter(cfg.TEST.TEST_DATA).load_dataset(None)
    output_dir = cfg.TEST.OUTPUT_DIR
    seeta_dataset_dir = os.path.join(output_dir, cfg.TEST.OUTPUT_NAME)
    if not os.path.exists(seeta_dataset_dir):
        os.makedirs(seeta_dataset_dir)
    output_data = new_dataset(seeta_dataset, seeta_dataset_dir)
    total = seeta_dataset.record_count()
    for i, record in enumerate(seeta_dataset.read()):
        print('Progress: %s / %s' % (i, total))
        origin_image = record['content']
        record["object"] = []
        image = cv2.imdecode(np.fromstring(origin_image, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            output_data.write(record)
            continue
        image, bboxes = ctpn(sess, net, image)
        for bbox in bboxes:
            record['object'].append({
                "xmin": int(bbox[0]),
                "ymin": int(bbox[1]),
                "xmax": int(bbox[2]),
                "ymax": int(bbox[3]),
                "score": round(bbox[-1], 3),
                "difficult": 0,
                "name": "sign",
            })
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
                "uuid": "seeta_dataset"
            }],
        }, fo)
