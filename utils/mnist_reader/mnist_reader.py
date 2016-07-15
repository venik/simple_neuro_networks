#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import matplotlib.cm as cm

import struct

# train data set
fname_train_dataset = "../../data_set/mnist/train-images-idx3-ubyte";
fd_tds = open(fname_train_dataset, 'rb');
train_head = struct.unpack('>IIII', fd_tds.read(16))

print("magic number:%d num of images:%d num of rows:%d num of columns:%d\n" % (train_head[0], train_head[1], train_head[2], train_head[3]) );

#data = bytearray(fd_tds.read(784));
#print(data.__len__())
#print("[" + data + "]")

#data = np.fromstring(fd_tds.read(784)).reshape((28, 28))
#data = np.array(fd_tds.read(784), np.uint8)

fd_tds.seek(784 * 33, 1)

data = np.fromfile(fd_tds, dtype=np.uint8, count=28 * 28)
array = np.reshape(data, [28, 28], order='C')

plt.imshow(array, cmap = cm.Greys_r)
plt.show()