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
"""Dataset loaders."""
import os
import glob
import time
import numpy as np
import mindspore as ms
from mindspore.dataset import vision, NumpySlicesDataset
from mindspore.dataset.vision import Inter
from mindvision.dataset import Mnist
from src.utils.data_preprocess import filter_36, inver_label, remove_contradicting, binary_image, flatten_image
from PIL import Image


def create_loaders(cfg):
    """Dataset loader"""
    if cfg.DATASET.type == "mnist":
        train_dataset, eval_dataset = create_mnist_loaders(cfg)
    elif cfg.DATASET.type == "warship":
        train_dataset, eval_dataset = create_warship_loaders(cfg)
    else:
        raise ValueError("cfg.DATASET.type must be mnist or warship")
    return train_dataset, eval_dataset


def create_mnist_loaders(config):
    # Load mnist data set (training set, eval set)
    train_data = Mnist(
        path=config.DATASET.path,
        split='train',
        shuffle=False,
        download=True
    )
    test_data = Mnist(
        path=config.DATASET.path,
        split='test',
        shuffle=False,
        download=True
    )
    train_dataset = train_data.dataset
    eval_dataset = test_data.dataset

    # Scale and normalize the image pixels
    train_dataset = train_dataset.map(vision.Rescale(1.0 / 255.0, 0), input_columns='image')
    eval_dataset = eval_dataset.map(vision.Rescale(1.0 / 255.0, 0), input_columns='image')

    # Filter numbers 3 and 6
    train_dataset = train_dataset.filter(predicate=filter_36, input_columns=['label'])
    eval_dataset = eval_dataset.filter(predicate=filter_36, input_columns=['label'])

    # Convert (3,6) labels to (0,1)
    train_dataset = train_dataset.map(operations=[inver_label], input_columns=['label'])
    eval_dataset = eval_dataset.map(operations=[inver_label], input_columns=['label'])

    # Bilinear interpolation shrinks the image size to 4 * 4
    train_dataset = train_dataset.map(operations=vision.Resize([4, 4], Inter.BILINEAR), input_columns='image')
    eval_dataset = eval_dataset.map(operations=vision.Resize([4, 4], Inter.BILINEAR), input_columns='image')

    # Remove contradictory data caused by shrinking pictures
    train_dataset = remove_contradicting(train_dataset)
    eval_dataset = remove_contradicting(eval_dataset)

    # Image binarization processing
    train_dataset = train_dataset.map(operations=[binary_image], input_columns='image')
    eval_dataset = eval_dataset.map(operations=[binary_image], input_columns='image')

    # Remove contradictory data caused by binarization
    train_dataset = remove_contradicting(train_dataset)
    eval_dataset = remove_contradicting(eval_dataset)

    # Save data preprocessed results
    if not os.path.exists(config.DATASET.path + 'pretreatment/'):
        os.system("mkdir -p " + config.DATASET.path + 'pretreatment/')
        train_dataset.save(config.DATASET.path + 'pretreatment/train_dataset')
        eval_dataset.save(config.DATASET.path + 'pretreatment/eval_dataset')
    train_dataset, eval_dataset = 0, 1
    return train_dataset, eval_dataset


def create_warship_loaders(config):
    # Load warship data set (training set, eval set)
    images = []
    for f in glob.iglob(config.DATASET.path + "Burke/*"):
        images.append(np.asarray(Image.open(f).convert('L')))
    for f in glob.iglob(config.DATASET.path + "Nimitz/*"):
        images.append(np.asarray(Image.open(f).convert('L')))
    images_test = []
    for f in glob.iglob(config.DATASET.path + "test_burke/*"):
        images_test.append(np.asarray(Image.open(f).convert('L')))
    for f in glob.iglob(config.DATASET.path + "test_nimitz/*"):
        images_test.append(np.asarray(Image.open(f).convert('L')))
    # Scale and normalize the image pixels
    images = np.array(images)
    images = images / 255
    images_test = np.array(images_test)
    images_test = images_test / 255
    # Generate label, Burke is 1, Nimitz is 0
    train_label = np.array([])
    test_label = np.array([])
    for i in range(202):
        i += 1
        i -= 1
        train_label = np.append(train_label, 1)
    for i in range(209):
        i += 1
        i -= 1
        train_label = np.append(train_label, 0)
    for i in range(78):
        i += 1
        i -= 1
        test_label = np.append(test_label, 1)
    for i in range(72):
        i += 1
        i -= 1
        test_label = np.append(test_label, 0)
    # Generate dataset from images and tags
    train_dataset = NumpySlicesDataset({'features': images, 'labels': train_label}, shuffle=False)
    eval_dataset = NumpySlicesDataset({'features': images_test, 'labels': test_label}, shuffle=False)
    # Bilinear interpolation reduces the image size to 4 * 4
    train_dataset = train_dataset.map(operations=vision.Resize([4, 4], Inter.BILINEAR), input_columns='features')
    eval_dataset = eval_dataset.map(operations=vision.Resize([4, 4], Inter.BILINEAR), input_columns='features')
    train_dataset = remove_contradicting(train_dataset)
    eval_dataset = remove_contradicting(eval_dataset)
    train_dataset = train_dataset.map(operations=[flatten_image], input_columns='features')
    eval_dataset = eval_dataset.map(operations=[flatten_image], input_columns='features')
    train_dataset = remove_contradicting(train_dataset)
    eval_dataset = remove_contradicting(eval_dataset)
    # shuffle dataset
    ms.dataset.config.set_seed(1234)
    train_dataset = train_dataset.shuffle(train_dataset.get_dataset_size())
    eval_dataset = eval_dataset.shuffle(eval_dataset.get_dataset_size())
    ms.dataset.config.set_seed(int(time.time()))
    # Save data preprocessed results
    if not os.path.exists(config.DATASET.path + 'pretreatment/'):
        os.system("mkdir -p " + config.DATASET.path + 'pretreatment/')
        train_dataset.save(config.DATASET.path + 'pretreatment/train_dataset')
        eval_dataset.save(config.DATASET.path + 'pretreatment/eval_dataset')
    train_dataset, eval_dataset = 0, 1
    return train_dataset, eval_dataset
