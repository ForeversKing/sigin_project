import os
import numpy as np
import math
import cv2
import logging
import xml.etree.ElementTree as ET


logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        datefmt='%d %b %Y %H:%M:%S',
        filename='./question_image_log.txt',
        filemode='a+')


def load_pascal_annoataion(p):
    '''
    load annotation from the text file
    :param p:
    :return:
    '''
    text_polys = []
    if not os.path.exists(p):
        return np.array(text_polys, dtype=np.float32)

    tree = ET.parse(p)
    anno = tree.getroot()
    objs = anno.findall('object')
    for obj in objs:
        rbox = obj.find("bndbox")
        xmin = float(rbox.find('xmin').text)
        xmax = float(rbox.find('xmax').text)
        ymin = float(rbox.find('ymin').text)
        ymax = float(rbox.find('ymax').text)
        assert xmax - xmin >= 1 and ymax - ymin >= 1, print('unreasonable xml:', p)
        text_polys.append([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    return text_polys


if __name__ == '__main__':
    path = '/home/yulongwu/d/data/sign_data/detection_data/JPEGImages'
    gt_path = '/home/yulongwu/d/data/sign_data/detection_data/Annotations'
    out_image_path = '/home/yulongwu/d/ocr/text-detection-ctpn/data/TEXTVOC/VOC2007/re_data/image'
    out_map_path = '/home/yulongwu/d/ocr/text-detection-ctpn/data/TEXTVOC/VOC2007/re_data/txt'
    if not os.path.exists(out_image_path):
        os.makedirs(out_image_path)

    if not os.path.exists(out_map_path):
        os.makedirs(out_map_path)
    files = os.listdir(path)
    files.sort()
    count = 0
    for file in files:
        _, basename = os.path.split(file)
        stem, ext = os.path.splitext(basename)
        img_path = os.path.join(path, file)
        gt_file = os.path.join(gt_path, stem + '.xml')
        if not os.path.exists(gt_file):
            continue
        if os.stat(img_path).st_size == 0:
            print('SKIPPING', img_path)
            logging.info('SKIPPING', img_path)
            continue
        img = cv2.imread(img_path)
        img_size = img.shape
        im_size_min = np.min(img_size[0:2])
        im_size_max = np.max(img_size[0:2])

        im_scale = float(600) / float(im_size_min)
        if np.round(im_scale * im_size_max) > 1200:
            im_scale = float(1200) / float(im_size_max)
        re_im = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        re_size = re_im.shape
        assert re_size[0] <= 1200 and re_size[1] <= 1200, print('the unreasonable image:', img_path)
        cv2.imwrite(os.path.join(out_image_path, basename), re_im)
        try:
            lines = load_pascal_annoataion(gt_file)
        except:
            print('question 0 xml file:', gt_file)
            logging.info('question xml file:', gt_file)
            continue
        if len(lines) == 0:
            print('question 1 xml file:', gt_file)
            continue
        # print('the image:', count)
        count += 1
        for line in lines:
            # splitted_line = line.strip().lower().split(',')
            pt_x = np.zeros((4, 1))
            pt_y = np.zeros((4, 1))
            pt_x[0, 0] = int(float(line[0][0]) / img_size[1] * re_size[1])
            pt_y[0, 0] = int(float(line[0][1]) / img_size[0] * re_size[0])
            pt_x[1, 0] = int(float(line[1][0]) / img_size[1] * re_size[1])
            pt_y[1, 0] = int(float(line[1][1]) / img_size[0] * re_size[0])
            pt_x[2, 0] = int(float(line[2][0]) / img_size[1] * re_size[1])
            pt_y[2, 0] = int(float(line[2][1]) / img_size[0] * re_size[0])
            pt_x[3, 0] = int(float(line[3][0]) / img_size[1] * re_size[1])
            pt_y[3, 0] = int(float(line[3][1]) / img_size[0] * re_size[0])

            ind_x = np.argsort(pt_x, axis=0)
            pt_x = pt_x[ind_x]
            pt_y = pt_y[ind_x]

            if pt_y[0] < pt_y[1]:
                pt1 = (pt_x[0], pt_y[0])
                pt3 = (pt_x[1], pt_y[1])
            else:
                pt1 = (pt_x[1], pt_y[1])
                pt3 = (pt_x[0], pt_y[0])

            if pt_y[2] < pt_y[3]:
                pt2 = (pt_x[2], pt_y[2])
                pt4 = (pt_x[3], pt_y[3])
            else:
                pt2 = (pt_x[3], pt_y[3])
                pt4 = (pt_x[2], pt_y[2])

            xmin = int(min(pt1[0], pt2[0]))
            ymin = int(min(pt1[1], pt2[1]))
            xmax = int(max(pt2[0], pt4[0]))
            ymax = int(max(pt3[1], pt4[1]))

            if xmin < 0:
                xmin = 0
            if xmax > re_size[1] - 1:
                xmax = re_size[1] - 1
            if ymin < 0:
                ymin = 0
            if ymax > re_size[0] - 1:
                ymax = re_size[0] - 1

            width = xmax - xmin
            height = ymax - ymin

            # reimplement
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
                x_right.append(x_left[i] + step-1)
            x_right.append(xmax)
            x_right = np.array(x_right)

            idx = np.where(x_left == x_right)
            x_left = np.delete(x_left, idx, axis=0)
            x_right = np.delete(x_right, idx, axis=0)
            txt_path = os.path.join(out_map_path, stem + '.txt')
            with open(txt_path, 'a') as f:
                for i in range(len(x_left)):
                    f.writelines("text\t")
                    f.writelines(str(int(x_left[i])))
                    f.writelines("\t")
                    f.writelines(str(int(ymin)))
                    f.writelines("\t")
                    f.writelines(str(int(x_right[i])))
                    f.writelines("\t")
                    f.writelines(str(int(ymax)))
                    f.writelines("\n")
