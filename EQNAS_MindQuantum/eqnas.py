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
"""python eqnas.py"""
import argparse
import time
import os
import src.utils.logger as lg
from src.utils.config import cfg
from src.dataset import create_loaders
from src.qea import qea
from src.utils.train_utils import train_qnn_final
from src.model.common import create_qnn

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    train_start = time.time()
    parser = argparse.ArgumentParser(description='eqnas parser')
    parser.add_argument('--data-type', type=str, default="mnist", help='mnist dataset type')
    parser.add_argument('--data-path', type=str, default="./dataset/mnist/", help='mnist dataset path')
    parser.add_argument('--batch', type=int, default=32, help='train batch size')
    parser.add_argument('--epoch', type=int, default=3, help='train epoch')
    parser.add_argument('--final', type=int, default=10, help='final train epoch')
    args = parser.parse_args()
    cfg.DATASET.type = args.data_type
    cfg.DATASET.path = args.data_path
    cfg.TRAIN.checkpoint_path = "./weights/" + cfg.DATASET.type + "/final/"
    cfg.TRAIN.BATCH_SIZE = args.batch
    cfg.TRAIN.EPOCHS = args.epoch  # 10 for warship
    cfg.TRAIN.EPOCHS_FINAL = args.final  # 20 for warship
    cfg.LOG_NAME = "train_" + cfg.DATASET.type
    cfg.ROOT = os.path.dirname(__file__)
    os.chdir(cfg.ROOT)

    logger = lg.get_logger(cfg.LOG_NAME)
    create_loaders(cfg)
    qea()
    logger.info("-------------------------final-------------")
    logger.info("best_acc: %s", str(cfg.QEA.best_acc))
    logger.info("best_arch: %s", str(cfg.QEA.best_arch))
    best_acc_list = cfg.QEA.best_acc.tolist()
    max_index = best_acc_list.index(max(best_acc_list))
    logger.info('best model is gen_indiv[%s][%s]', str(max_index), str(cfg.QEA.best_chrom[max_index]))
    best_accuracy = train_qnn_final(cfg.QEA.best_arch[max_index], max_index, cfg.QEA.best_chrom[max_index],
                                    cfg.TRAIN.EPOCHS_FINAL)
    logger.info('best accuracy : %s with quantum circuit :', str(best_accuracy))
    logger.info(create_qnn(cfg.QEA.best_arch[max_index]))
    complete = time.time() - train_start
    lg_info = 'Training complete in ({:.0f}h {:.0f}m {:.0f}s)'.format(complete // 3600, complete // 60, complete % 60)
    logger.info(lg_info)
    total_epochs = cfg.TRAIN.EPOCHS * cfg.QEA.N * cfg.QEA.generation_max + cfg.TRAIN.EPOCHS_FINAL
    steps = cfg.TRAIN.DATA_SIZE * total_epochs
    speed = (complete * 1000) / steps
    lg_info = "{:.0f}ms/step on average".format(speed)
    logger.info(lg_info)
