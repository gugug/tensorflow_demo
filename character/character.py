# coding=utf-8
# ==============================================================================

"""Functions for reading data."""

import gzip
import os
import numpy
import math

from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile

import base

CHARACTER_LABELS = {'适应性': 0, '社交性': 1, '开放性': 2, '利他性': 3, '道德感': 4}


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gfile.Open(filename, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                             (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def combine_blog_to_one(blog_dir):
    """combine the blog content under blog_dir into one txt"""
    files = gfile.ListDirectory(blog_dir)
    print(len(files))
    for f in files:
        one_dir = os.path.join(blog_dir, f)
        blog_files = gfile.ListDirectory(one_dir)
        content_file = open(os.path.join(one_dir, "all_content.txt"), "w+")
        for bf in blog_files:
            filename = os.path.join(one_dir, bf)
            content = extract_content(filename)
            content_file.write(content)
        content_file.close()


def extract_faw(train_dir):
    """
    提取特征数据
    提取train下info.txt的性别，粉丝，关注，微博
    :param train_dir:
    :return:
    """
    files = gfile.ListDirectory(train_dir)
    train_data_file = open(os.path.join(train_dir, "train_data.txt"), "w+")
    for userid in files:
        infofilepath = os.path.join(train_dir, userid, 'info.txt')
        faw_list = extract_info_faw(infofilepath)
        train_data_file.write(userid + " ")
        for faw in faw_list:
            train_data_file.write(str(faw) + " ")
        train_data_file.write('\n')
    train_data_file.close()


def extract_info_faw(filename):
    """ load info.txt"""
    with gfile.Open(filename, 'rb') as gf:
        lines = gf.readlines()
        sex = lines[2].replace("\n", '').split(":")[1]
        if sex == '男':
            sex = 1
        elif sex == '女':
            sex = -1
        else:
            sex = 0
        fans = lines[-1].replace("\n", '').split(":")[1]
        atten = lines[-2].replace("\n", '').split(":")[1]
        weibo = lines[-3].replace("\n", '').split(":")[1]
        return [sex, math.log(int(fans) + 1), math.log(int(atten) + 1), math.log(int(weibo) + 1)]


def extract_content(filename):
    """
    return content in filename
    :param filename:
    :return:
    """
    with gfile.Open(filename, 'rb') as gf:
        lines = gf.readlines()
        return lines[1]


def extract_labels(filename):
    """Extract the labels into a 1D uint8 numpy array [value]."""
    print('Extracting', filename)
    with gfile.Open(filename, 'rb') as f:
        lines = f.readlines()
        ids = []
        labels = []
        print('labels size are ', len(lines))
        for line in lines:
            line = line.replace('\n', '')
            id_character = line.split('\t\t\t\t\t\t')
            ids.append(id_character[0])
            print(id_character[1].split('、')[0])  # 性格
            print(CHARACTER_LABELS.get(id_character[1].split('、')[0]))  # 性格标签
            labels.append(CHARACTER_LABELS.get(id_character[1].split('、')[0]))
            # print("%s user's character is %s" % (id_character[0], id_character[1]))
        # print(labels)
        # print(numpy.array(ids).__getitem__(1))
        # print numpy.array(ids)
        # print numpy.array(labels)
        write_labels(filename + '.txt', ids, labels)
        return numpy.array(ids), numpy.array(labels)


def write_labels(filename, ids, labels):
    with gfile.Open(filename, 'w+') as f:
        for idx in range(len(ids)):
            f.write(ids[idx] + " " + str(labels[idx]))
            f.write('\n')


def parallel_train_data_label(datafilename, labelname):
    """parallel the data and labels"""
    id_label_dict = {}
    with gfile.Open(labelname, 'rb') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace("\n", "")
            id_label = line.split(" ")
            id_label_dict.setdefault(id_label[0], id_label[1])
    train_txt = open('train_data_label.txt', 'w+')
    with gfile.Open(datafilename, 'rb') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            id = line.split(' ')[0]
            train_txt.write(line + str(id_label_dict.get(id)))
            train_txt.write('\n')
    train_txt.close()


def load_data_label(filename):
    """load data and label"""
    xs = []
    ys = []
    with gfile.Open(filename, 'rb') as f:
        lines = f.readlines()
        for line in lines:
            if len(line) > 0:
                line = line.replace("\n", "")
                id_label = line.split(" ")
                # print id_label[0],(id_label[-1])
                ys.append(int(id_label[-1]))
                xs.append([float(id_label[1]), float(id_label[2]), float(id_label[3]), float(id_label[4])])
                # xs.append([float(id_label[1])])


    VALIDATION_SIZE = 30
    validate_data = numpy.array(xs[:VALIDATION_SIZE])
    validate_data_label = numpy.array(ys[:VALIDATION_SIZE])
    train_data = numpy.array(xs[VALIDATION_SIZE:])
    train_data_label = numpy.array(ys[VALIDATION_SIZE:])
    return validate_data, validate_data_label, train_data, train_data_label


if __name__ == '__main__':
    base_path = '/media/gu/493ce6af-560c-45cb-9cc4-66119285f579/gu/PycharmProjects/character_data/all_data'

    # extract_faw(os.path.join(base_path,'user'))


    # extract_labels(os.path.join(base_path,' all_user_character_map'))

    parallel_train_data_label(os.path.join(base_path, 'train_data.txt'),
                              os.path.join(base_path, 'all_user_character_map.txt'))
    load_data_label(os.path.join(base_path, 'train_data_label.txt'))


class DataSet(object):
    def __init__(self,
                 images,
                 labels,
                 fake_data=False,
                 one_hot=False,
                 dtype=dtypes.float32):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
            self._num_examples = images.shape[0]

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2])
            if dtype == dtypes.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                images = images.astype(numpy.float32)
                images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)
                ]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir,
                   fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32):
    if fake_data:
        def fake():
            return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype)

        train = fake()
        validation = fake()
        test = fake()
        return base.Datasets(train=train, validation=validation, test=test)

    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
    VALIDATION_SIZE = 5000

    local_file = base.maybe_download(TRAIN_IMAGES, train_dir)
    train_images = extract_images(local_file)

    local_file = base.maybe_download(TRAIN_LABELS, train_dir)
    train_labels = extract_labels(local_file, one_hot=one_hot)

    local_file = base.maybe_download(TEST_IMAGES, train_dir)
    test_images = extract_images(local_file)

    local_file = base.maybe_download(TEST_LABELS, train_dir)
    test_labels = extract_labels(local_file, one_hot=one_hot)

    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]

    train = DataSet(train_images, train_labels, dtype=dtype)
    validation = DataSet(validation_images, validation_labels, dtype=dtype)
    test = DataSet(test_images, test_labels, dtype=dtype)

    return base.Datasets(train=train, validation=validation, test=test)
