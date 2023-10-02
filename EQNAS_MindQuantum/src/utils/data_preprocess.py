# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Data preprocess functions
"""
import collections
from src.utils.config import cfg


def filter_36(label):
    return (label == 3) | (label == 6)


# convert 3/6 to 0/1
def inver_label(label):
    if label == 3:
        return 0
    return 1


# Remove contradictory data caused by shrinking pictures
def remove_contradicting(dataset):
    mapping = collections.defaultdict(set)  # Dictionary for storing (picture hashable key, set (labels)) key value pairs
    data_iter = dataset.create_tuple_iterator()
    for image, label in data_iter:
        key = tuple(image.asnumpy().flatten())  # Convert image to a tuple of hashable
        mapping[key].add(int(label.asnumpy()))

    def filter_contradict(image):
        key = tuple(image.flatten())
        if len(mapping[key]) == 2:
            return False
        return True

    if cfg.DATASET.type == "mnist":
        return dataset.filter(predicate=filter_contradict, input_columns=['image'])
    return dataset.filter(predicate=filter_contradict, input_columns=['features'])


def binary_image(image):
    return (image > cfg.DATASET.THRESHOLD).astype(int).flatten()


def flatten_image(image):
    return image.flatten()
