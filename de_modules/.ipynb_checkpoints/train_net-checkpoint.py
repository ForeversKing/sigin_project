import pprint
import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from lib.fast_rcnn.train import SolverWrapper
from lib.fast_rcnn.config import cfg_from_file
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg
import seetaas_helper as helper


if __name__ == '__main__':
    yaml_cfg = helper.get_parameter(yaml_file="adl.yml")
    cfg_from_file(yaml_cfg)
    print('Using config:')
    pprint.pprint(cfg)
    device_name = '/gpu:0'
    print(device_name)
    train_graph = tf.Graph()
    with train_graph.as_default():
        network = get_network('VGGnet_train')
        solver = SolverWrapper(train_graph, network,
                               output_dir='/tmp/model',
                               dataset_path='/tmp/dataset',
                               dataset_name='root')
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        with tf.Session(config=config, graph=train_graph) as sess:
            solver.train_model(sess)
