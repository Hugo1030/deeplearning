# -*- coding: utf-8 -*-

import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

file = ['corpus/lena512.png']
file_queue = tf.train.string_input_producer(file)

reader = tf.WholeFileReader()
key,value = reader.read(file_queue)
images = tf.image.decode_png(value, channels=1)

# build CNN model
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    images = tf.image.convert_image_dtype(images, dtype=tf.float32)
    before_image = sess.run(images)
    print('before_image:', before_image.shape)

    # get input_
    input_ = tf.expand_dims(before_image, 0)
    print("input_:", input_)

    # get filter
    filter_ = np.asarray([[1/9, 1/9, 1/9],
                         [1/9, 1/9, 1/9],
                         [1/9, 1/9, 1/9]], dtype=np.float32)
    filter_ = tf.expand_dims(filter_, -1)
    filter_ = tf.expand_dims(filter_, -1)
    print("filter_:", filter_)

    # finished CNN
    output = tf.nn.conv2d(input_, filter_, strides=[1, 1, 1, 1], padding='SAME')
    output_value = sess.run(output)
    after_image = output_value[0, :, :, :]
    print('after_image:', after_image.shape)

    # close read
    coord.request_stop()
    coord.join(threads)

# draw image
plt.imshow(before_image[:, :, 0], cmap='gray')
plt.show()
plt.imshow(after_image[:, :, 0], cmap='gray')
plt.show()

