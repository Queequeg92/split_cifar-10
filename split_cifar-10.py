# Copyright 2017 Queequeg92.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import numpy as np
from skimage import io

# Path to cifar-10 dataset. Only binary version is supported.
# Download url: https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
path = '/path/to/cifar-10/binary/dataset'

# Labels.
label_strings = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
collection =[]

# Split train dataset.
for i in range(1, 6):
    fpath = os.path.join(path, 'data_batch_' + str(i) + '.bin')
    raw = np.fromfile(fpath, dtype='uint8')
    collection.append(np.reshape(raw, (10000, 3073)))
records = np.concatenate(collection, axis=0)
labels = records[:, 0]
images = np.reshape(records[:, 1:], (50000, 3, 32, 32,))
# Gather different classes to corresponding files.
os.makedirs('data/train')
for i in range(10):
    index = (labels == i)
    class_labels = labels[index]
    class_images = images[index]
    class_records = np.concatenate(
        [np.expand_dims(class_labels, axis=1), np.reshape(class_images, (5000, -1))],
        axis=1)
    raw_records = class_records.flatten()
    class_file_name = 'data/train/' + label_strings[i] + '.bin'
    raw_records.tofile(class_file_name)


# Split test dataset.
fpath = os.path.join(path, 'test_batch.bin')
raw = np.fromfile(fpath, dtype='uint8')
records = np.reshape(raw, (10000, 3073))
labels = records[:, 0]
images = np.reshape(records[:, 1:], (10000, 3, 32, 32,))
# Gather different classes to corresponding files.
os.makedirs('data/test')
for i in range(10):
    index = (labels == i)
    class_labels = labels[index]
    class_images = images[index]
    class_records = np.concatenate(
        [np.expand_dims(class_labels, axis=1), np.reshape(class_images, (1000, -1))],
        axis=1)
    raw_records = class_records.flatten()
    class_file_name = 'data/test/' + label_strings[i] + '.bin'
    raw_records.tofile(class_file_name)
