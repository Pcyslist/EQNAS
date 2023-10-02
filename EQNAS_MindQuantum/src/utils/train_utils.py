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
"""
The implementation of training qnn
"""
import logging
import pickle
import os
import glob
import mindspore as ms
import mindspore.nn as nn
import numpy as np
from mindvision.engine.callback import ValAccMonitor
from src.loss import CustomizedBCELoss
from src.model.common import create_qnn
from src.metrics import CustomizedBCEAcc
from src.utils.config import cfg


def train_qnn(chromosome, generation, indiv_num, EPOCHS):
    logger = logging.getLogger(cfg.LOG_NAME)
    train_dataset = ms.dataset.MindDataset(cfg.DATASET.path + 'pretreatment/train_dataset').batch(cfg.TRAIN.BATCH_SIZE,
                                                                                                  drop_remainder=True)
    eval_dataset = ms.dataset.MindDataset(cfg.DATASET.path + 'pretreatment/eval_dataset').batch(cfg.TRAIN.BATCH_SIZE,
                                                                                                drop_remainder=True)
    cfg.TRAIN.DATA_SIZE = train_dataset.get_dataset_size()
    lg_info = f"Dataset size: {cfg.TRAIN.DATA_SIZE}"
    logger.info(lg_info)
    qnn = create_qnn(chromosome)
    if not os.path.exists("./weights/"):
        os.system("mkdir ./weights")
    with open('./weights/model.arch', 'wb') as f:
        pickle.dump(chromosome, f)
    ms.save_checkpoint(save_obj=qnn, ckpt_file_name='./weights/init.ckpt')
    loss = CustomizedBCELoss()
    opti = nn.Adam(qnn.trainable_params(), learning_rate=cfg.TRAIN.learning_rate)
    model = ms.Model(network=qnn, loss_fn=loss, optimizer=opti, metrics={'Accuracy': CustomizedBCEAcc()})
    # training strategy
    loss_monitor = ms.LossMonitor(train_dataset.get_dataset_size() // 5)  # 一个 epoch 输出5次监测到的损失
    acc_monitor = ValAccMonitor(model=model, dataset_val=eval_dataset, num_epochs=EPOCHS, metric_name='Accuracy',
                                ckpt_directory='./weights', dataset_sink_mode=False)
    model.train(epoch=EPOCHS, train_dataset=train_dataset, callbacks=[loss_monitor, acc_monitor],
                dataset_sink_mode=False)
    # save ckpt
    ms.save_checkpoint(qnn, './weights/latest.ckpt')
    # load best ckpt for inference
    weights = ms.load_checkpoint(net=qnn, ckpt_file_name='./weights/best.ckpt')
    load_finished = ms.load_param_into_net(net=qnn, parameter_dict=weights)
    if not load_finished:
        logger.info('Best model loading success!')
        logger.info('The best model parameters :')
        logger.info(qnn.parameters_dict()['ansatz_weight'].asnumpy())
    else:
        logger.info('Best model loading error!')

    # model eval
    model_eval = ms.Model(network=qnn, loss_fn=loss, optimizer=opti, metrics={'Accuracy': CustomizedBCEAcc()})
    accuracy = model_eval.eval(eval_dataset, dataset_sink_mode=False)['Accuracy']
    logger.info('Eval accuracy : %s', str(accuracy))
    os.system("mkdir -p ./weights/gen_{}/ind_{}acc_{}%".format(generation, indiv_num, np.round(accuracy, 10) * 100))
    os.system("mv ./weights/model.arch ./weights/init.ckpt ./weights/best.ckpt ./weights/latest.ckpt "
              "./weights/gen_{}/ind_{}acc_{}%".format(generation, indiv_num, np.round(accuracy, 10) * 100))
    del model, train_dataset, eval_dataset, model_eval
    return accuracy


def train_qnn_final(chromosome, generation, indiv_num, EPOCHS):
    logger = logging.getLogger(cfg.LOG_NAME)
    train_dataset = ms.dataset.MindDataset(cfg.DATASET.path + 'pretreatment/train_dataset').batch(cfg.TRAIN.BATCH_SIZE,
                                                                                                  drop_remainder=True)
    eval_dataset = ms.dataset.MindDataset(cfg.DATASET.path + 'pretreatment/eval_dataset').batch(cfg.TRAIN.BATCH_SIZE,
                                                                                                drop_remainder=True)
    lg_info = f"Dataset size: {cfg.TRAIN.DATA_SIZE}"
    logger.info(lg_info)
    qnn = create_qnn(chromosome)
    with open('./weights/model.arch', 'wb') as f:
        pickle.dump(chromosome, f)
    dir_init = "./weights/gen_{}/ind_{}*".format(generation, int(indiv_num))
    logger.info(glob.glob(dir_init))
    assert len(glob.glob(dir_init)) == 1
    dir_init = glob.glob(dir_init)[0]
    weights = ms.load_checkpoint(ckpt_file_name=dir_init + '/init.ckpt', net=qnn)
    ms.load_param_into_net(parameter_dict=weights, net=qnn)
    ms.save_checkpoint(save_obj=qnn, ckpt_file_name='./weights/init.ckpt')
    loss = CustomizedBCELoss()
    opti = nn.Adam(qnn.trainable_params(), learning_rate=cfg.TRAIN.learning_rate)
    model = ms.Model(network=qnn, loss_fn=loss, optimizer=opti, metrics={'Accuracy': CustomizedBCEAcc()})
    # train strategy
    loss_monitor = ms.LossMonitor(train_dataset.get_dataset_size() // 5)  # 一个 epoch 输出5次监测到的损失
    acc_monitor = ValAccMonitor(model=model, dataset_val=eval_dataset, num_epochs=EPOCHS, metric_name='Accuracy',
                                ckpt_directory='./weights', dataset_sink_mode=False)
    model.train(epoch=EPOCHS, train_dataset=train_dataset, callbacks=[loss_monitor, acc_monitor],
                dataset_sink_mode=False)
    # save last ckpt
    ms.save_checkpoint(qnn, './weights/latest.ckpt')
    # load best ckpt for inference and eval
    weights = ms.load_checkpoint(net=qnn, ckpt_file_name='./weights/best.ckpt')
    load_finished = ms.load_param_into_net(net=qnn, parameter_dict=weights)
    if not load_finished:
        logger.info('Best model loading success!')
        logger.info('The best model parameters :')
        logger.info(qnn.parameters_dict()['ansatz_weight'].asnumpy())
    else:
        logger.info('Best model loading error!')

    # model eval
    model_eval = ms.Model(network=qnn, loss_fn=loss, optimizer=opti, metrics={'Accuracy': CustomizedBCEAcc()})
    accuracy = model_eval.eval(eval_dataset, dataset_sink_mode=False)['Accuracy']
    logger.info('Eval accuracy : %s', str(accuracy))
    os.system(
        "mkdir -p ./weights/final/gen_{}/ind_{}acc_{}%".format(generation, indiv_num, np.round(accuracy, 10) * 100))
    os.system("mv ./weights/model.arch ./weights/init.ckpt ./weights/best.ckpt ./weights/latest.ckpt "
              "./weights/final/gen_{}/ind_{}acc_{}%".format(generation, indiv_num, np.round(accuracy, 10) * 100))
    if not os.path.exists(cfg.TRAIN.checkpoint_path):
        os.system("mkdir -p " + cfg.TRAIN.checkpoint_path)
    os.system("cp -r ./weights/final/gen_{}/ind_{}acc_{}%/* ".format(generation, indiv_num, np.round(accuracy, 10)
                                                                     * 100) + cfg.TRAIN.checkpoint_path)
    del model, train_dataset, eval_dataset, model_eval
    return accuracy
