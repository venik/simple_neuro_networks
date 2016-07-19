#!/usr/bin/env python
# GPLv2 License
__author__ = 'Sasha Nikiforov nikiforov.al[at]gmail.com'

########
# Reader for the dataset from Yann LeCun website
# http://yann.lecun.com/exdb/mnist/index.html
import struct
import numpy as np

class mnist(object):
    _isTrainMode = False
    _fname_dataset = ""
    _fname_labels = ""
    _last_frame = 0
    _frames_in_dataset = 0
    _labels_in_dataset = 0

    # contstants. all details on the Yann's site
    _data_header_size = 16
    _frame_size = 784   # 28 by 28
    _data_header_pattern = '>IIII'

    _label_header_size = 8
    _label_size = 1   # 1 byte - uint8
    _label_header_pattern = ">II"

    _fd_tds = None
    _fd_tls = None

    def __init__(self, mnist_folder, isTrainMode = True):
        self._isTrainMode = isTrainMode
        if (self._isTrainMode == True):
            self._fname_dataset = mnist_folder + "/train-images-idx3-ubyte"
            self._fname_labels = mnist_folder + "/train-labels-idx1-ubyte"
        else:
            self._fname_dataset = mnist_folder + "/t10k-images-idx3-ubyte"
            self._fname_labels = mnist_folder + "/t10k-labels-idx1-ubyte"

        self._last_frame = 0

        self._fd_tds = open(self._fname_dataset, 'rb')
        # just skip file header, all details on the Yann's site
        # format (magic_number, num_of_frames, rows, columns)
        (_, self._frames_in_dataset, _, _) = struct.unpack(self._data_header_pattern, self._fd_tds.read(self._data_header_size))

        self._fd_tls = open(self._fname_labels, 'rb')
        (_, self._labels_in_dataset) = struct.unpack(self._label_header_pattern, self._fd_tls.read(self._label_header_size))


    def get_arbitrary_frame(self, frame_id):
        # assert((self._fd_tds == None) or (self._fd_tls == None), "mnist data set were teared down")

        self._last_frame = frame_id

        # first parameter: header + size_of_frame * frame_id
        # second parameter: SEEK_SET or 0
        self._fd_tds.seek(self._data_header_size + self._frame_size * self._last_frame, 0)
        self._fd_tls.seek(self._label_header_size + self._label_size * self._last_frame, 0)

        (label, data) = self.get_next_frame()
        return (label, data)

    def get_next_frame(self):
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

    def get_num_of_frames_in_dataset(self):
        return self._frames_in_dataset