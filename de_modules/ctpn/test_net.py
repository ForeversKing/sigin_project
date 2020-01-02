from __future__ import print_function

import cv2
import glob
import os
import shutil
import sys
import pickle
import numpy as np
import tensorflow as tf

sys.path.append(os.getcwd())
from lib.networks.factory import get_network
from lib.datasets.factory import get_imdb
from lib.fast_rcnn.config import cfg, get_output_dir, cfg_from_file
from lib.fast_rcnn.test import test_ctpn
from lib.utils.timer import Timer
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg

os.environ['CUDA_VISIBLE_DEVICES'] = ' '

CLASSES = ('__background__', 'text')


def check_unreasonable_box(boxes, scale):
    boxes = list(boxes)
    boxes_lst = []
    boxes_tmp = []
    l = len(boxes)
    for i in range(l):
        if i in boxes_tmp:
            continue
        if i == l-1:
            continue
        for j in range(i+1, l):
            delt = []
            for index in [0, 1, 6, 7]:
                delt.append(boxes[i][index] - boxes[j][index])
            if delt[0] <=0 and delt[1] <=0 and delt[2] >=0 and delt[3] >=0:
                print('fuck1')
                boxes_tmp.append(j)
            if delt[0] >=0 and delt[1] >=0 and delt[2] <=0 and delt[3] <=0:
                print('fuck2')
                boxes_tmp.append(i)
    for i in range(l):
        if i in boxes_tmp:
            continue
        boxes_copy = []
        if np.linalg.norm(boxes[i][0] - boxes[i][2]) < 5 or np.linalg.norm(boxes[i][1] - boxes[i][7]) < 5:
            continue
        min_x = min(int(boxes[i][0] / scale), int(boxes[i][2] / scale), int(boxes[i][4] / scale), int(boxes[i][6] / scale))
        min_y = min(int(boxes[i][1] / scale), int(boxes[i][3] / scale), int(boxes[i][5] / scale), int(boxes[i][7] / scale))
        max_x = max(int(boxes[i][0] / scale), int(boxes[i][2] / scale), int(boxes[i][4] / scale), int(boxes[i][6] / scale))
        max_y = max(int(boxes[i][1] / scale), int(boxes[i][3] / scale), int(boxes[i][5] / scale), int(boxes[i][7] / scale))
        # for ii in range(len(boxes[i])):
        #     if ii in [0, 1, 6, 7, 8]:
        #         boxes_copy.append(boxes[i][ii])
        boxes_lst.append([min_x, min_y, max_x, max_y, boxes[i][-1]])
    return boxes_lst


def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f


def draw_boxes(img, image_name, boxes, scale):
    base_name = image_name.split('/')[-1]
    with open('../data/LikeVOC/eval_result/' + 'res_{}.txt'.format(base_name.split('.')[0]), 'w') as f:
        for box in boxes:
            if np.linalg.norm(box[0] - box[2]) < 5 or np.linalg.norm(box[1] - box[7]) < 5:
                continue
            if box[8] >= 0.9:
                color = (0, 255, 0)
            elif box[8] >= 0.8:
                color = (255, 0, 0)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
            cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)

            min_x = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            min_y = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
            max_x = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            max_y = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))

            line = ','.join([str(min_x), str(min_y), str(max_x), str(max_y)]) + '\r\n'
            f.write(line)

    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join("../data/LikeVOC/eval_result", base_name), img)


def test_net(sess, net, imdb, weights_filename):
    timer = Timer()
    timer.tic()
    np.random.seed(cfg.RNG_SEED)
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    output_dir = get_output_dir(imdb, weights_filename)
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    # all_boxes = []
    all_boxes = [[[] for _ in range(imdb.num_classes)]
                 for _ in range(num_images)]
    print(all_boxes)
    for i in range(num_images):
        print('***********', imdb.image_path_at(i))
        img = cv2.imread(imdb.image_path_at(i))
        img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
        scores, boxes = test_ctpn(sess, net, img)
        textdetector = TextDetector()
        boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
        print(('Detection took {:.3f}s for '
               '{:d} object proposals').format(timer.total_time, boxes.shape[0]))
        boxes = check_unreasonable_box(boxes, scale)
        all_boxes[i][1] += boxes
    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    imdb.evaluate_detections(all_boxes, output_dir)
    timer.toc()


if __name__ == '__main__':
    if os.path.exists("../data/LikeVOC/eval_result/"):
        shutil.rmtree("../data/LikeVOC/eval_result/")
    os.makedirs("../data/LikeVOC/eval_result/")

    cfg_from_file('ctpn/text.yml')

    imdb = get_imdb('voc_2007_test')
    imdb.competition_mode(False)

    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # load network
    net = get_network("VGGnet_test")
    # load model
    print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
    saver = tf.train.Saver()

    try:
        ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
        model = ckpt.model_checkpoint_path
        filename = os.path.splitext(os.path.basename(model))[0]
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

    image_path = '/home/yulongwu/d/ocr/jinshi_ocr_data/test_image.lst'
    test_net(sess, net, imdb, filename)
