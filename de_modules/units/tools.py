import cv2
from math import sqrt
import numpy as np
import random


def standardization(image):
    h, w = image.shape[:2]
    mean = np.mean(image)
    dev = np.std(image)
    num_elements = h * w * 3
    adjusted_stddev = max(dev, 1.0 / sqrt(num_elements))
    return (image - mean) / adjusted_stddev


def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f


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


def calculate_paramter(bboxes, label):
    B = (bboxes[3] - bboxes[1])* 48.828125
    Fc = ((bboxes[3] - bboxes[1])/2+ bboxes[1])*48.828125
    t1 = 0.00128*bboxes[0]
    t2 = 0.00128*bboxes[2]
    T = t2-t1
    return [label, B, Fc, t1, t2, T]


def cv_vis(im, dets, labels):
    colors = [(0, 255, 0), (255, 255, 0), (0, 255, 255)]
    for i, (det, label) in enumerate(zip(dets, labels)):
        color = random.choice(colors)
        im = cv2.rectangle(im, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), color, 2)
        label, B, Fc, t1, t2, T = calculate_paramter(det, label)
        im = cv2.putText(im, 'label: %s' % label, (int((det[0] + det[2]) / 2), int(det[1]+10)),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
        im = cv2.putText(im, 'B: %g Hz' % B, (int((det[0] + det[2]) / 2), int(det[1]+30)),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
        im = cv2.putText(im, 'Fc: %g Hz' % Fc, (int((det[0] + det[2]) / 2), int(det[1]+50)),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
        im = cv2.putText(im, 't1: %g s' % t1, (int((det[0] + det[2]) / 2), int(det[1]+70)),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
        im = cv2.putText(im, 't2: %g s' % t2, (int((det[0] + det[2]) / 2), int(det[1]+90)),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
        im = cv2.putText(im, 'T: %g s' % T, (int((det[0] + det[2]) / 2), int(det[1]+110)),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)


    return im


def constrate_image(images, labels, bboxes):
    target_images = np.zeros((1024*len(images), 1024, 3), dtype=np.uint8)
    for i, (label, bbox) in enumerate(zip(labels, bboxes)):
        cv2.imwrite('/tmp/image/%s.png' % i, images[i])
        image = cv_vis(images[i], bbox, label)
        target_images[i*1024:(i+1)*1024, :, :] = image
    return target_images


def resize_boxes(boxes, scale):
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
    return text_boxes


def seg_image(image, target_h):
    origin_h, origin_w = image.shape[:2]
    print(origin_h)
    # assert origin_h % target_h == 0
    target_images = []
    for i in range(0, origin_h, target_h):
        if i+target_h <= origin_h:
            im = image[i:i+target_h, :, :].astype(np.float32)
            assert im.shape[0] == im.shape[1] == 1024
            target_images.append(im)
    print('>>>>>>>>>>>', len(target_images))
    return target_images


def get_image_batch(images, height, width):
    images_list = []
    for image in images:
        im = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        im.astype(np.float32)
        im = standardization(im)
        images_list.append(im[None, ...])
    return images_list


def _slice(lst, inds):
    slices = []
    k = 0
    for ind in inds:
        slice = []
        for i in range(ind):
            slice.append(lst[i+k])
        slices.append(slice)
        k += ind
    return slices


def get_rec_image(image_list, bbox_list):
    assert len(image_list) == len(bbox_list)
    sub_images = []
    for i, (image, bboxes) in enumerate(zip(image_list, bbox_list)):
        if len(bboxes) <0:
            continue
        for bbox in bboxes:
            sub_images.append(image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :])
    return sub_images


