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
"""python eval.py"""
import pickle
import os
import argparse
import mindspore as ms
import src.utils.logger as lg
from src.loss import CustomizedBCELoss
from src.metrics import CustomizedBCEAcc
from src.model.common import create_qnn
from src.utils.config import cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='eqnas parser')
    parser.add_argument('--data-type', type=str, default="mnist", help='mnist dataset type')
    parser.add_argument('--data-path', type=str, default="./dataset/mnist/", help='mnist dataset path')
    parser.add_argument('--ckpt-path', type=str, default="./weights/mnist/final/", help='mnist model ckpt path')
    args = parser.parse_args()
    cfg.DATASET.type = args.data_type
    cfg.DATASET.path = args.data_path
    cfg.TRAIN.checkpoint_path = args.ckpt_path
    cfg.LOG_NAME = "eval_" + cfg.DATASET.type
    cfg.ROOT = os.path.dirname(__file__)
    os.chdir(cfg.ROOT)

    logger = lg.get_logger(cfg.LOG_NAME)
    eval_dataset = ms.dataset.MindDataset(cfg.DATASET.path + 'pretreatment/eval_dataset')
    eval_dataset = eval_dataset.batch(cfg.TRAIN.BATCH_SIZE)
    it = eval_dataset.create_tuple_iterator()
    image, label = next(it)
    logger.info("input size: %s", str(image.shape))
    with open(cfg.TRAIN.checkpoint_path + 'model.arch', 'rb') as f:
        arch = pickle.load(f)
    qnn = create_qnn(arch)
    weights_best = ms.load_checkpoint(net=qnn, ckpt_file_name=cfg.TRAIN.checkpoint_path + 'best.ckpt')
    ms.load_param_into_net(net=qnn, parameter_dict=weights_best)
    model = ms.Model(network=qnn, loss_fn=CustomizedBCELoss(), metrics={'Accuracy': CustomizedBCEAcc()})
    logger.info("eval running...")
    logger.info(model.eval(eval_dataset, dataset_sink_mode=False))
