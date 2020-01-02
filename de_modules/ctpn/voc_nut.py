# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob

###
def change_class(cls):
    return cls.split('_')[0]
###

def compute_ap(recall, precision):
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    # mrec = [0.  0.2 0.4 0.4 0.4 0.4 0.6 0.8 0.8 0.8 1.  1. ]
    # mpre = [0.   1.   1.   0.67 0.5  0.4  0.5  0.57 0.5  0.44 0.5  0.  ]

    # compute the precision envelope(计算精度包络线，也就是计算precision曲线的外轮廓)
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    # mpre = [1.   1.   1.   0.67 0.57 0.57 0.57 0.57 0.5  0.5  0.5  0.  ]

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]
    # i = [0 1 5 6 9]
    # mrec[1:] = [0.2 0.4 0.4 0.4 0.4 0.6 0.8 0.8 0.8 1.  1. ]
    # mrec[:-1] = [0.  0.2 0.4 0.4 0.4 0.4 0.6 0.8 0.8 0.8 1. ]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


class Eval():
    def __init__(self, use_diff=False):
        self._classes = ('__background__', 'text')
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': use_diff,
                       'matlab_eval': False,
                       'rpn_file': None}
        self._devkit_path = '/home/yulongwu/d/data/sign_data/test_data20191108'
        self._image_set = None
        self._year = 2012

    def parse_rec(self, filename):
        """ Parse a PASCAL VOC xml file """
        tree = ET.parse(filename)
        objects = []
        for obj in tree.findall('object'):
            obj_struct = {}
            obj_struct['name'] = obj.find('name').text
            # obj_struct['name'] = change_class(obj.find('name').text)
            # obj_struct['pose'] = obj.find('pose').text
            # obj_struct['truncated'] = int(obj.find('truncated').text)
            obj_struct['difficult'] = int(obj.find('difficult').text)
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(float(bbox.find('xmin').text)),
                                  int(float(bbox.find('ymin').text)),
                                  int(float(bbox.find('xmax').text)),
                                  int(float(bbox.find('ymax').text))]
            objects.append(obj_struct)

        return objects

    def voc_ap(self, rec, prec, use_07_metric=False):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:False).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def voc_eval(self, detpath,
                 annopath_root,
                 classname,
                 cachedir=None,
                 ovthresh=0.5,
                 use_07_metric=False,
                 use_diff=False):
        """rec, prec, ap = voc_eval(detpath,
                                    annopath,
                                    imagesetfile,
                                    classname,
                                    [ovthresh],
                                    [use_07_metric])

        Top level function that does the PASCAL VOC evaluation.

        detpath: Path to detections
            detpath.format(classname) should produce the detection results file.
        annopath: Path to annotations
            annopath.format(imagename) should be the xml annotations file.
        imagesetfile: Text file containing the list of images, one image per line.
        classname: Category name (duh)
        cachedir: Directory for caching the annotations
        [ovthresh]: Overlap threshold (default = 0.5)
        [use_07_metric]: Whether to use VOC07's 11 point AP computation
            (default False)
        """
        # assumes detections are in detpath.format(classname)
        # assumes annotations are in annopath.format(imagename)
        # assumes imagesetfile is a text file with each line an image name
        # cachedir caches the annotations in a pickle file

        # first load gt
        if not os.path.isdir(cachedir):
            os.mkdir(cachedir)
        cachefile = os.path.join(cachedir, 'cache_annots.pkl')
        if not os.path.isfile(cachefile):
            recs = {}
            annopaths = glob.glob(os.path.join(annopath_root, '*'))
            for annopath in annopaths:
                imagename = os.path.basename(annopath).split('.')[0]
                recs[imagename] = self.parse_rec(annopath)
            with open(cachefile, 'wb') as f:
                pickle.dump(recs, f)
        else:
            # load
            if os.path.getsize(cachefile) > 0:
                with open(cachefile, 'rb') as f:
                    try:
                        recs = pickle.load(f)
                    except:
                        recs = pickle.load(f, encoding='bytes')
            else:
                print('the file is empty!!!!!!')
        print(recs)
        # extract gt objects for this class
        class_recs = {}
        npos = 0
        for imagename in recs.keys():
            R = [obj for obj in recs[imagename] if change_class(obj['name']) == classname]
            # R = [obj for obj in recs[imagename] if obj['name'] == classname]
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[imagename] = {'bbox': bbox,
                                     'difficult': difficult,
                                     'det': det}
        # read dets
        detfile = detpath
        with open(detfile, 'r') as f:
            lines = f.readlines()

        splitlines = [x.strip().split(',') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

        ap = self.voc_ap(rec, prec, use_07_metric)
        # self.plot_voc(fp, rec)
        return rec, prec, ap

    def plot_voc(self, fp, rec):
        fig = plt.figure()
        plt.title('[10,35][35,90][1.5.2.5]-iter_2000_train_nut')
        plt.xlabel('FP')
        plt.ylabel('RECALL')
        ax = plt.subplot(111)
        ax.scatter(fp.tolist(), rec.tolist(), marker='.')
        plt.ylim(0, 1)
        plt.show()

    def _do_python_eval(self, output_dir='output'):
        annopath_root = os.path.join(
            self._devkit_path,
            'new_Annotations',
            )
        cachedir = os.path.join(self._devkit_path, 'new_Annotations')
        aps = []

        # The PASCAL VOC metric changed in 2010new_Annotations
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            recs = []
            precs = []
            # filename = self._get_voc_results_file_template().format(cls)
            filenames = glob.glob(os.path.join(self._devkit_path, 'pre_annotations', 'total.txt'))
            for filename in filenames:
                use_07_metric = True if int(self._year) < 2010 else False
                print('~~~~~~~~~~~~~~~~', filename)
                # print('~~~~~~~~~~~~~~~~q', cls)
                # print('~~~~~~~~~~~~~~~~s', cachedir)
                rec, prec, ap = self.voc_eval(
                    filename, annopath_root, cls, cachedir, ovthresh=0.5,
                    use_07_metric=use_07_metric, use_diff=self.config['use_diff'])
                print(rec)
                recs.extend(list(rec))
                precs.extend(list(prec))
                aps.append(ap)
            AP = compute_ap(recs, precs)
            print('##########AP', AP)
            recs = np.array(recs)
            precs = np.array(precs)
            # print(recs)
            # print(precs)
            # print('>>>>>', np.mean(recs), np.mean(precs))
            # print(('AP for {} = {:.4f}'.format(cls, ap)))
            # with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            #     pickle.dump({'rec': np.mean(recs), 'prec': np.mean(precs), 'ap': ap}, f)
        print(('Mean AP = {:.4f}'.format(np.mean(aps))))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print(('{:.3f}'.format(ap)))
        print(('{:.3f}'.format(np.mean(aps))))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')


if __name__ == '__main__':
    voc_eval = Eval()
    voc_eval._do_python_eval()