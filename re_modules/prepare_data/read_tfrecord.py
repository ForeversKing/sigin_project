import os
import tensorflow as tf


def _get_data_queue(base_dir, file_patterns=['*.tfrecord'], capacity=2 ** 15,
                    num_epochs=None):
    """Get a data queue for a list of record files"""
    # List of lists ...
    data_files = [tf.gfile.Glob(os.path.join(base_dir, file_pattern))
                  for file_pattern in file_patterns]
    # flatten
    data_files = [data_file for sublist in data_files for data_file in sublist]
    data_queue = tf.train.string_input_producer(data_files,
                                                capacity=capacity,
                                                num_epochs=num_epochs)
    return data_queue


def _read_word_record(data_queue, batch_size, num_threads=4):
    reader = tf.TFRecordReader()  # Construct a general reader
    key, example_serialized = reader.read(data_queue)

    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/labels': tf.FixedLenFeature([1], dtype=tf.int64),
        'image/width': tf.FixedLenFeature([1], dtype=tf.int64,
                                          default_value=1),
        'image/height': tf.FixedLenFeature([1], dtype=tf.int64,
                                           default_value=1)
    }
    features = tf.parse_single_example(example_serialized, feature_map)

    width = tf.cast(features['image/width'], tf.int32)
    height = tf.cast(features['image/height'], tf.int32)
    # image = tf.image.decode_jpeg(features['image/encoded'],
    #                              channels=3)
    image = tf.decode_raw(features['image/encoded'], tf.uint8)
    image = tf.reshape(image, [64, 256, 3])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.per_image_standardization(image)

    label = tf.cast(features['image/labels'], tf.int32)  # for batching

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=num_threads,
                                              capacity=num_threads * batch_size * 2)
    return image_batch, label_batch


def read_and_decode(base_dir, batch_size, num_threads):
    data_queue = _get_data_queue(base_dir, capacity=batch_size * num_threads)
    image_batch, label_batch = _read_word_record(data_queue, batch_size, num_threads=num_threads)
    return image_batch, label_batch


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    data = read_and_decode('/home/yulongwu/d/data/sign_data/recognize_data/0/tf_record', 24, num_threads=4)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for _ in range(200):
            image_batch, label_batch = sess.run(data)
            print(label_batch.shape)

        coord.request_stop()
        coord.join(threads)
