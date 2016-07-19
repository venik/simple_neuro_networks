#!/usr/bin/env python

# simple test for the lib.mnist.mnist class
# GPLv2 License
__author__ = 'Sasha Nikiforov nikiforov.al[at]gmail.com'

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
sys.path.append('../../')
from lib.mnist.mnist import *

data_set = mnist('../../data_set/mnist')

[first_label, first_data] = data_set.get_next_frame()
[arbitrary_label, arbitrary_data] = data_set.get_arbitrary_frame(frame_id = 784)
data_set.tear_down()

print("Last frame id: " + str(data_set.get_num_of_frames_in_dataset()))

print("arbitrary label: " + str(arbitrary_label))
arbitrary_array = np.reshape(arbitrary_data, [28, 28], order='C')
plt.imshow(arbitrary_array, cmap = cm.Greys_r)
plt.show()

print("first label: " + str(first_label))
first_array = np.reshape(first_data, [28, 28], order='C')
plt.imshow(first_array, cmap = cm.Greys_r)
plt.show()