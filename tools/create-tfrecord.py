import time
import os
import hashlib

from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import lxml.etree
import tqdm

flags.DEFINE_string('data_dir', './data/voc2012_raw/VOCdevkit/VOC2012/',
                    'path to raw PASCAL VOC dataset')
flags.DEFINE_enum('split', 'train', [
    'train', 'val'], 'specify train or val spit')
flags.DEFINE_string(
    'output_file', './data/train.tfrecord', 'output dataset')
# flags.DEFINE_string('classes', './data/voc2012.names', 'classes file')
flags.DEFINE_string('classes', './data/cervix-colpo.names', 'classes file')

def build_example(annotation, class_map):
    img_path = os.path.join(
        FLAGS.data_dir, 'JPEGImages', annotation['filename'])
    img_raw = open(img_path, 'rb').read()
    key = hashlib.sha256(img_raw).hexdigest()

    width = int(annotation['size']['width'])
    height = int(annotation['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    if 'object' in annotation:
        for obj in annotation['object']:
            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(
                float(obj['bndbox']['xmax']) / width)
            ymax.append(
                float(obj['bndbox']['ymax']) / height)

            classes_text.append(obj['name'].encode('utf8'))
            classes.append(class_map[obj['name']])

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),

        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            annotation['filename'].encode('utf8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            annotation['filename'].encode('utf8')])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),

        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),

        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
    }))

    return example

def main(_argv):
    class_map = {name: idx for idx, name in enumerate(
        open(FLAGS.classes).read().splitlines())}
    logging.info("Class mapping loaded: %s", class_map)

    # make tfrecord
    writer = tf.io.TFRecordWriter(FLAGS.output_file)
    # get IMAGE files
    image_list = open(os.path.join(
        FLAGS.data_dir, 'ImageSets', 'Main', '%s.txt' % FLAGS.split)).read().splitlines()
    logging.info("Image list loaded: %d", len(image_list))
    for name in tqdm.tqdm(image_list):
        # get JSON
        annotation = json.load(open(os.path.join(image_dir, name)))
        tf_example = build_example(
            annotation, class_map)
        writer.write(
            tf_example.SerializeToString())
        writer.close()


logging.info("Done")


if __name__ == '__main__':
    app.run(main)
