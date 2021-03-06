'''
Usage

1. full model
python3 detect.py --classes ./data/cervix-colpo.names \
--image ./data/cervix/colpo/C_C0032_SS_NIA11_032_L.png \
--notiny --weights ./checkpoints/yolov3_train_2.tf \
--output ./output.jpg --num_classes 5

2. tiny model
python3 detect.py --classes ./data/cervix-colpo.names \
--image ./data/cervix/colpo/C_C0032_SS_NIA11_032_L.png \
--tiny --weights ./checkpoints/yolov3_train_2.tf \
--output ./output.jpg --num_classes 5

3. tfrecord
python3 detect.py --classes ./data/cervix-colpo.names \
--tfrecord ./data/colpo-test.tfrecord \
--notiny --weights ./checkpoints/yolov3_train_2.tf \
--output ./output --num_classes 5

4. a image file
python3 detect.py --classes ./data/cervix-colpo.names \
--image ./data/cervix/colpo/C_C0032_SS_NIA11_032_L.png \
--notiny --weights ./checkpoints/yolov3_train_2.tf \
--output ./output.jpg --num_classes 5
'''

import os
import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', './output.jpg', 'path to output dir or image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_enum('log_level', 'info', [
    'info', 'debug'], 'log_level mode; debug/info')


def _detect(yolo, img_raw, img_size):
    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, img_size)

    t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))

    return boxes, scores, classes, nums


def _save_result_img(filename, img_raw, boxes, scores, classes, nums, class_names):
    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)

    cv2.imwrite(filename, img)
    logging.info('output saved to: {}'.format(filename))


def _print_results(nums, class_names, classes, scores, boxes):
    for i in range(nums[0]):
        logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                           np.array(scores[0][i]),
                                           np.array(boxes[0][i])))


def main(_argv):
    # log level
    logging.debug("FLAGS.log_level: {}".format(FLAGS.log_level))
    if 'debug' == FLAGS.log_level:
        logging.set_verbosity(logging.DEBUG)
    else:
        logging.set_verbosity(logging.INFO)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    if FLAGS.tfrecord:
        logging.debug('tfrecord is {}'.format(FLAGS.tfrecord))
        dataset = load_tfrecord_dataset(
            FLAGS.tfrecord, FLAGS.classes, FLAGS.size)
        dataset = dataset.shuffle(512)

        if not os.path.exists(FLAGS.output):
            os.makedirs(FLAGS.output)

        for step, (img_raw, _label) in enumerate(dataset):
            # detect
            logging.info("step: {}".format(step))
            boxes, scores, classes, nums = _detect(yolo, img_raw, FLAGS.size)
            # save
            output_file = os.path.join(FLAGS.output, ("output_" + str(step) + ".jpg"))
            _save_result_img(output_file, img_raw, boxes, scores, classes, nums, class_names)
            # print info
            _print_results(nums, class_names, classes, scores, boxes)
    else:
        img_raw = tf.image.decode_image(
            open(FLAGS.image, 'rb').read(), channels=3)
        # detect
        boxes, scores, classes, nums = _detect(yolo, img_raw, FLAGS.size)
        output_file = FLAGS.output
        # save
        _save_result_img(output_file, img_raw, boxes, scores, classes, nums, class_names)
        # print info
        _print_results(nums, class_names, classes, scores, boxes)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
