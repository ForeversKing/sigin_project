from __future__ import print_function

import cv2
import glob
import os
import shutil
import sys
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.fast_rcnn.test import test_ctpn
from lib.utils.timer import Timer
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def calculateIoU(candidateBound, groundTruthBound):
    cx1 = candidateBound[0]
    cy1 = candidateBound[1]
    cx2 = candidateBound[2]
    cy2 = candidateBound[3]

    gx1 = groundTruthBound[0]
    gy1 = groundTruthBound[1]
    gx2 = groundTruthBound[2]
    gy2 = groundTruthBound[3]

    carea = (cx2 - cx1) * (cy2 - cy1) #C的面积
    garea = (gx2 - gx1) * (gy2 - gy1) #G的面积

    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = w * h #C∩G的面积

    iou_candidate = area / carea
    iou_groundtruth = area / garea

    return iou_candidate, iou_groundtruth


def tmp_contrate(images):
    master_image = np.zeros((10*1024, 1024, 3), dtype=np.uint8)
    for i in range(len(images)):
        image = cv2.resize(images[i], (1024, 1024))
        master_image[i*512:(i+1)*512, :, :] = image[:512, :, :]
    return master_image


def process_boxes(texts_boxes):
    texts_boxes = np.array(texts_boxes)
    tmp_texts_boxes = texts_boxes.copy()
    number_boxes = tmp_texts_boxes.shape[0]
    remove_ixs = []
    for i in range(number_boxes - 1):
        for j in range(i + 1, number_boxes):
            w_i = tmp_texts_boxes[i][2]-tmp_texts_boxes[i][0]
            w_j = tmp_texts_boxes[j][2]-tmp_texts_boxes[j][0]
            iou_i, iou_j = calculateIoU(tmp_texts_boxes[i], tmp_texts_boxes[j])
            if iou_i != iou_j:
                if iou_i <= 0.1 and iou_j <=0.1:
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


def vis_detection(im, dets, save_path):
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    colors = ['blue']
    for det in dets:
        color = random.choice(colors)
        bbox = det[:4]
        score = det[-1]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor=color, linewidth=2.0)
        )
        ax.text(int((bbox[0]+bbox[2])/2), bbox[1] - 2,
                '{:.3f}'.format(score),
                bbox=dict(facecolor=color, alpha=0.5),
                fontsize=14, color=color)
    print('+++++++', im.shape)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.savefig(save_path, dpi=20)
    plt.show()


def cv_vis(im, dets, save_path):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    colors = [(0, 255, 0), (255, 255, 0), (0, 255, 255)]
    for det in dets:
        color = random.choice(colors)
        im = cv2.rectangle(im, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), color, 2)
        score = det[4]
        # cls = det[-1]
        im = cv2.putText(im, '{:.3f}'.format(score), (int((det[0]+det[2])/2), int((det[3]+det[1])/2)),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(save_path, im)
    cv2.imshow(' ', im)
    cv2.waitKey()


def draw_boxes(img, image_name, boxes, scale):
    base_name = image_name.split('/')[-1]
    text_boxes = []
    # with open('/home/yulongwu/d/data/sign_data/test_data20191108/pre_Annotations/' + 'res_{}.txt'.format(base_name.split('.')[0]), 'w') as f:
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
    lines = []
    for boxes in text_boxes:
        print(boxes)
        line = ','.join([str(int(boxes[0])), str(int(boxes[1])), str(int(boxes[2])), str(int(boxes[3]))]) + '\n'
        lines.append(base_name.split('.')[0] + ',' + str(boxes[-1]) + ',' + line)
        # f.write(base_name.split('.')[0] + ',' + str(boxes[-1]) + ',' + line)
    save_path = os.path.join("/home/yulongwu/d/ocr/text-detection-ctpn/data/ctpn_demo_result/", base_name)
    # vis_detection(img, text_boxes, save_path)
    cv_vis(img, text_boxes, save_path)
    return lines


def ctpn(sess, net, image_name):
    timer = Timer()
    timer.tic()
    img = cv2.imread(image_name)
    img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    scores, boxes = test_ctpn(sess, net, img)
    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    sort_index = np.argsort(boxes[:, -1])[::-1]
    boxes = boxes[sort_index]
    # print(boxes)
    texts = draw_boxes(img, image_name, boxes, scale)
    timer.toc()
    print(('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0]))
    return texts


if __name__ == '__main__':
    cfg_from_file('./text.yml')
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    net = get_network("VGGnet_test")
    # load model
    print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
    saver = tf.train.Saver()

    try:
        ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

    im_names = glob.glob(os.path.join('/tmp/image', '11.png'))
    with open('/home/yulongwu/d/data/sign_data/test_data20191108/pre_annotations/total.txt', 'w') as fi:
        for im_name in im_names:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print(('Demo for {:s}'.format(im_name)))
            lines = ctpn(sess, net, im_name)
            for line in lines:
                fi.write(line)

