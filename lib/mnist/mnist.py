#!/usr/bin/env python
# GPLv2 License
__author__ = 'Sasha Nikiforov nikiforov.al[at]gmail.com'

########
# Reader for the dataset from Yann LeCun website
# http://yann.lecun.com/exdb/mnist/index.html
import struct
import numpy as np

class mnist(object):
    _fname_train_dataset = ""
    _fname_labels_dataset = ""
    _last_frame = 0

    # contstants. all details on the Yann's site
    _data_header_size = 16
    _frame_size = 784   # 28 by 28
    _data_header_pattern = '>IIII'

    _label_header_size = 8
    _label_size = 1   # 1 byte - uint8
    _label_header_pattern = ">II"

    _fd_tds = None
    _fd_tls = None

    def __init__(self, mnist_folder):
        self._fname_train_dataset = mnist_folder + "/train-images-idx3-ubyte"
        self._fname_labels_dataset = mnist_folder + "/train-labels-idx1-ubyte"

        self._last_frame = 0

        self._fd_tds = open(self._fname_train_dataset, 'rb')
        # just skip file header, all details on the Yann's site
        train_head = struct.unpack(self._data_header_pattern, self._fd_tds.read(self._data_header_size))
        # print("magic number:%d num of images:%d num of rows:%d num of columns:%d" %
        #       (train_head[0], train_head[1], train_head[2], train_head[3]) )

        self._fd_tls = open(self._fname_labels_dataset, 'rb')
        labels_head = struct.unpack(self._label_header_pattern, self._fd_tls.read(self._label_header_size))
        # print("magic number:%d num of images:%d" %
        #      (labels_head[0], labels_head[1]) )

    def get_arbitrary_train_frame(self, frame_id):
        # assert((self._fd_tds == None) or (self._fd_tls == None), "mnist data set were teared down")

        self._last_frame = frame_id

        # first parameter: header + size_of_frame * frame_id
        # second parameter: SEEK_SET or 0
        self._fd_tds.seek(self._data_header_size + self._frame_size * self._last_frame, 0)
        self._fd_tls.seek(self._label_header_size + self._label_size * self._last_frame, 0)

        (label, data) = self.get_next_train_frame()
        return (label, data)

    def get_next_train_frame(self):
        data = np.fromfile(self._fd_tds, dtype=np.uint8, count=self._frame_size)
        label = np.fromfile(self._fd_tls, dtype=np.uint8, count=self._label_size)

        self._last_frame = self._last_frame + 1

        return (label, data)

    def tear_down(self):
        # train data set
        if self._fd_tds != None:
            self._fd_tds.close()
            self._fd_tds = None

        if self._fd_tls != None:
            self._fd_tls.close()
            self._fd_tls = None