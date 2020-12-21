'''
Usage
python tools/create-tfrecord.py train --data_dir=data/voc2012_raw/VOCdevkit/VOC2012/ \
    --output_file=data/train.tfrecord \
    --classes=data/voc2012.names

python tools/create-tfrecord.py train --data_dir=data/voc2012_raw/VOCdevkit/VOC2012/ \
    --output_file=data/train.tfrecord \
    --classes=data/cervix-colpo.names

python tools/create-tfrecord.py train --data_dir=data/cervix/colpo \
    --output_file=data/cervix-colpo-train.tfrecord \
    --classes=data/cervix-colpo.names --log_level=debug

python tools/create-tfrecord.py train --data_dir=data/cervix/colpo \
    --output_file=data/cervix-colpo-train.tfrecord \
    --classes=data/cervix-colpo.names --log_level=info
'''

import time
import os
import hashlib
import json

from os import listdir
from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import tqdm
import skimage.io


flags.DEFINE_string('data_dir', './data/voc2012_raw/VOCdevkit/VOC2012/',
                    'path to raw PASCAL VOC dataset')
flags.DEFINE_enum('split', 'train', [
    'train', 'val'], 'specify train or val spit')
flags.DEFINE_string(
    'output_file', './data/train.tfrecord', 'output dataset')
flags.DEFINE_string('classes', './data/voc2012.names', 'classes file')
flags.DEFINE_enum('log_level', 'info', [
    'info', 'debug'], 'log_level mode; debug/info')


def _get_height_width(img_path):
    image = skimage.io.imread(img_path)
    height, width = image.shape[:2]

    logging.debug('height, width: [%d %d]', height, width)

    return height, width


def _get_bbox(coordinates):
    xmin = 0
    ymin = 0
    xmax = 0
    ymax = 0

    if 0 == len(coordinates):
        logging.warning('input coordinate size is zero')
    else:
        all_point_x = []
        all_point_y = []
        for set in coordinates:
            for a in set:
                all_point_x.append(int(a[0]))
                all_point_y.append(int(a[1]))

        xmin = min(all_point_x)
        ymin = min(all_point_y)
        xmax = max(all_point_x)
        ymax = max(all_point_y)

    # logging.log(logging.DEBUG, 'bbox: [%d %d %d %d]', xmin, ymin, xmax, ymax)
    logging.debug('bbox: [%d %d %d %d]', xmin, ymin, xmax, ymax)

    return xmin, ymin, xmax, ymax


def build_example(annotation, path, filename, img_format, class_map):
    if img_format not in [".jpg", ".jpeg", ".png"]:
        logging.warning("image(%s) is not supperted format(%s)",
            filename, img_format)
        return None
    # img_path = os.path.join(FLAGS.data_dir, subset, filename)
    filename += img_format
    img_path = os.path.join(path, filename)

    img_raw = open(img_path, 'rb').read()
    key = hashlib.sha256(img_raw).hexdigest()

    height, width = _get_height_width(img_path)

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []

    logging.debug("path: %s", path)
    logging.debug("filename: %s", filename)

    for obj in annotation:
        if 'MultiPolygon' == obj['geometry']['type']:
            logging.warning("Multi-polygon type")
            return None

        _xmin, _ymin, _xmax, _ymax = _get_bbox(obj['geometry']['coordinates'])
        xmin.append(_xmin)
        ymin.append(_ymin)
        xmax.append(_xmax)
        ymax.append(_ymax)

        if 'classification' in obj['properties']:
            classes_text.append(obj['properties']['classification']['name']
                    .encode('utf8'))
            classes.append(
                class_map[obj['properties']['classification']['name']])
        else:
            classes_text.append(str('UNKNOWN').encode('utf8'))
            classes.append(class_map['UNKNOWN'])

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            filename.encode('utf8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            filename.encode('utf8')])),
        'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_format.encode('utf8')])),

        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),

        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
    }))

    return example

def main(_argv):
    logging.info("===== Start create tfrecord =====")
    # log level
    if 'debug' == FLAGS.log_level:
        logging.set_verbosity(logging.DEBUG)
    else:
        logging.set_verbosity(logging.INFO)

    # get class names
    class_map = {name: idx for idx, name in enumerate(
        open(FLAGS.classes).read().splitlines())}
    logging.info("Class mapping loaded: %s", class_map)

    # get IMAGE files
    base_dir = os.path.abspath("./")
    train_dir = os.path.join(base_dir, FLAGS.data_dir, 'train')
 
    image_list = [f for f in listdir(train_dir) 
        if f.endswith(".jpg") or f.endswith(".png")]

    logging.info("train path: %s", train_dir)
    logging.info("Image list loaded: %d", len(image_list))
    
    # open tfrecord file
    writer = tf.io.TFRecordWriter(FLAGS.output_file)
    for name in tqdm.tqdm(image_list):
        filename = os.path.splitext(name)[0]
        img_format = os.path.splitext(name)[1]
        json_file = filename + '.json'

        annotation = json.load(open(os.path.join(train_dir, json_file)))
        tf_example = build_example(annotation, train_dir, filename, 
            img_format, class_map)
        if tf_example is not None:  
            writer.write(tf_example.SerializeToString())

    writer.close()
    logging.info("===== Done =====")

    ## report??


if __name__ == '__main__':
    app.run(main)
