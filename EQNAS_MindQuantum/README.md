# 目录

<!-- TOC -->

- [目录](#目录)
- [EQNAS描述](#EQNAS描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [导出过程](#导出过程)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
            - [warship数据集上训练EQNAS](#warship数据集上训练EQNAS)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# EQNAS描述

## 概述

针对基于量子电路的量子神经网络，EQNAS提出了一种基于量子进化算法的神经网络结构搜索方法。在EQNAS方法中，通过搜索最佳网络结构，以提高模型精度，降低量子电路的复杂性，并减轻构建实际量子电路的负担，用于解决图像分类任务。

# 模型架构

![Qnn Structure](https://gitee.com/Pcyslist/cartographic-bed/raw/master/mnist_qnn.png)

该模型设计实现了一个量子神经网络用于图像分类，并基于量子进化算法进行神经架构搜索，网络结构主要包括两个模块：

- 量子编码线路Encoder：分别使用01编码和Rx编码对不同的数据集图片进行编码
- 待训练线路Ansatz：使用双比特量子门（*XX*门、*YY*门、*ZZ*门）以及量子 *I* 门构建了一个两层的量子神经网络Ansatz

通过对量子神经网络输出执行泡利 z 算符测量哈密顿期望，并利用量子进化算法对上述量子神经网络进行架构搜索，提高模型精度，降低线路复杂度。

# 数据集

使用的数据集：

- 数据集 [MNIST](<http://yann.lecun.com/exdb/mnist/>) 描述：MNIST数据集一共有7万张图片，其中6万张是训练集，1万张是测试集。每张图片是 28×28 的0 − 9的手写数字图片组成。每个图片是黑底白字的形式，黑底用0表示，白字用0-1之间的浮点数表示，越接近1，颜色越白。本模型筛选出其中的"3"和"6"类别，进行二分类。
- 数据集 [Warship](<https://gitee.com/Pcyslist/mqnn/blob/master/warship.zip>) 描述： 为了验证QNN对更复杂图像数据集的分类效果以及我们提出的EQNAS方法的有效性，我们采用了一组舰船目标数据集。该数据集是一艘航行中的船只，由无人机从不同角度拍摄。图像采用JPG格式，分辨率为640×512。它包含两个类别：Burke和Nimitz。该数据集的训练集数量为411（Burke类202个,Nimitz类209个），测试集数量为150（Burke类78个,Nimitz类72个）。

下载后，将数据集解压到如下目录：

  ```python
~/path/to/EQNAS/dataset/mnist
~/path/to/EQNAS/dataset/warship
  ```

# 环境要求

- 硬件（GPU）

    - 使用GPU来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
    - [MinQuantum](https://www.mindspore.cn/mindquantum/docs/en/r0.7/mindquantum_install.html)

- 其他第三方库安装

  ```bash
  cd EQNAS
  conda env create -f eqnas.yaml
  conda install --name eqnas --file condalist.txt
  pip install -r requirements.txt
  ```

- 如需查看详情，请参见如下资源：
  - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
  - [MindQuantum教程](https://www.mindspore.cn/mindquantum/docs/en/r0.7/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore以及MindQuantum后，您可以按照如下步骤进行训练和评估：

- GPU环境运行

  ```bash
  # 训练
  # mnist 数据集训练示例
  python eqnas.py --data-type mnist --data-path ./dataset/mnist/ --batch 32 --epoch 3 --final 10 | tee mnist_train.log
  OR
  bash run_train.sh mnist /abs_path/to/dataset/mnist/ 32 3 10
  # warship 数据集训练示例
  python eqnas.py --data-type warship --data-path ./dataset/warship/ --batch 10 --epoch 10 --final 20 | tee warship_train.log
  OR
  bash run_train.sh warship /abs_path/to/dataset/warship/ 10 10 20

  # 训练完成之后可执行评估
  # mnist数据集评估
  python eval.py --data-type mnist --data-path ./dataset/mnist/ --ckpt-path /abs_path/to/best_ckpt/ | tee mnist_eval.log
  OR
  bash run_eval.sh mnist /abs_path/to/dataset/mnist/ /abs_path/to/best_ckpt/
  # warship数据集评估
  python eval.py --data-type warship --data-path ./dataset/warship/ --ckpt-path /abs_path/to/best_ckpt/ | tee warship_eval.log
  OR
  bash run_eval.sh warship /abs_path/to/dataset/warship/ /abs_path/to/best_ckpt/
  ```

# 脚本说明

## 脚本及样例代码

```bash
├── EQNAS
    ├── condalist.txt                   # Anaconda env list
    ├── eqnas.py                        # 训练脚本
    ├── eqnas.yaml                      # Anaconda 环境配置
    ├── eval.py                         # 评估脚本
    ├── README.md                       # EQNAS模型相关说明
    ├── requirements.txt                # pip 包依赖
    ├── scripts
    │   ├── run_eval.sh                 # 评估shell脚本
    │   └── run_train.sh                # 训练shell脚本
    └── src
        ├── dataset.py                  # 数据集生成
        ├── loss.py                     # 模型损失函数
        ├── metrics.py                  # 模型评价指标
        ├── model
        │   └── common.py               # Qnn量子神经网络创建
        ├── qea.py                      # 量子进化算法
        └── utils
            ├── config.py               # 模型参数配置文件
            ├── data_preprocess.py      # 数据预处理
            ├── logger.py               # 日志构造器
            └── train_utils.py          # 模型训练定义
```

## 脚本参数

在config.py中可以同时配置量子进化算法参数、训练参数、数据集、和评估参数。

  ```python
  cfg = EasyDict()
  cfg.LOG_NAME = "logger"data_preprocess

  # Quantum evolution algorithm parameters
  cfg.QEA = EasyDict()
  cfg.QEA.fitness_best = []  # The best fitness of each generation

  # Various parameters of the population
  cfg.QEA.Genome = 64  # Chromosome length
  cfg.QEA.N = 10  # Population size
  cfg.QEA.generation_max = 50  # Population Iterations

  # Dataset parameters
  cfg.DATASET = EasyDict()
  cfg.DATASET.type = "mnist"  # mnist or warship
  cfg.DATASET.path = "./dataset/"+cfg.DATASET.type+"/"  # ./dataset/mnist/ or ./dataset/warship/
  cfg.DATASET.THRESHOLD = 0.5

  # Training parameters
  cfg.TRAIN = EasyDict()
  cfg.TRAIN.EPOCHS = 3  # 10 for warship
  cfg.TRAIN.EPOCHS_FINAL = 10  # 20 for warship
  cfg.TRAIN.BATCH_SIZE = 32  # 10 for warship
  cfg.TRAIN.learning_rate = 0.001
  cfg.TRAIN.checkpoint_path = "./weights/"+cfg.DATASET.type+"/final/"
  ```

更多配置细节请参考`utils`目录下config.py文件。

## 训练过程

### 训练

- GPU环境运行训练mnist数据集

  运行以下命令时请将数据集移动到EQNAS根目录下`dataset`文件夹下中，则可使用相对路径描述数据集位置，否则请将`--data-path`设置为绝对路径。

  ```bash
  python eqnas.py --data-type mnist --data-path ./dataset/mnist/ --batch 32 --epoch 3 --final 10 | tee mnist_train.log
  OR
  bash run_train.sh mnist /abs_path/to/dataset/mnist/ 32 3 10
  ```

  上述python命令将在后台运行，您可以通过当前目录下的`mnist_train.log`文件或者`./log/`目录下面的日志文件查看结果。

  训练结束后，您可在`eqnas.py`脚本所在目录下的`./weights/`目录下找到架构搜索过程中每一个模型对应的`best.ckpt、init.ckpt、latest.ckpt`文件以及`model.arch`模型架构文件。

- GPU环境运行训练warship数据集

  ```bash
  python eqnas.py --data-type warship --data-path ./dataset/warship/ --batch 10 --epoch 10 --final 20 | tee warship_train.log
  OR
  bash run_train.sh warship /abs_path/to/dataset/warship/ 10 10 20
  ```

  查看模型训练结果，与mnist数据集训练结果方式相同。

## 评估过程

### 评估

- 在GPU环境运行评估mnist数据集

- 在运行以下命令之前，清将数据集移动到EQNAS根目录下`dataset`文件夹下中，则可使用相对路径描述数据集位置，否则清给出数据集的绝对路径。

- 请用于评估的检查点路径。请将检查点路径设置为绝对路径。

  ```bash
  python eval.py --data-type mnist --data-path ./dataset/mnist/ --ckpt-path /abs_path/to/best_ckpt/ | tee mnist_eval.log
  OR
  bash run_eval.sh mnist /abs_path/to/dataset/mnist/ /abs_path/to/best_ckpt/
  ```

  上述python命令将在后台运行，您可以通过mnist_eval.log文件查看结果。

- 在GPU环境运行评估warship数据集

  请参考评估mnist数据集。

## 导出过程

### 导出MindIR

- 基于MindQuantum创建的量子模型，目前官方还不支持导出为该格式
- 但为了能够将量子线路进行保存，本项目中利用Python自带pickle数据序列化包，将架构搜索得到的每一个量子模型都保存为`./weights/model/model.arch`，您可以按照`eval.py`中的方法加载模型架构

# 模型描述

## 性能

### 训练性能

#### warship、mnist数据集上训练EQNAS

| 参数            | GPU                                           | GPU                                           |
| --------------- | --------------------------------------------- | --------------------------------------------- |
| 模型版本        | EQNAS                                         | EQNAS                                         |
| 资源            | NVIDIA GeForce RTX 3090 ；系统 ubuntu20.04    | NVIDIA GeForce RTX2080Ti ; 系统ubuntu18.04    |
| 上传日期        | 2022-12-6                                     | 2022-12-6                                     |
| MindSpore版本   | 1.8.1                                         | 1.8.1                                         |
| MindQuantum版本 | 0.7.0                                         | 0.7.0                                         |
| 数据集          | warship                                       | mnist                                         |
| 训练参数        | epoch=20, steps per epoch=41, batch_size = 10 | epoch=10.steps per epoch=116, batch_size = 32 |
| 优化器          | Adam                                          | Adam                                          |
| 损失函数        | Binary  CrossEntropy Loss                     | Binary  CrossEntropy Loss                     |
| 输出            | accuracy                                      | accuracy                                      |
| 精度            | 84.0%                                         | 98.9%                                         |
| 训练时长        | 7h19m29s                                      | 27h27m23s                                     |
| 速度            | 631毫秒/步                                    | 2734毫秒/步                                   |

# 随机情况说明

- 脚本dataset.py中，在创建舰船数据加载器时，对舰船数据进行打乱处理时，设置了随机数种子
- 为保证量子进化算法的变异、交叉操作的随机性，在上述设置随机数种子之后，随即以系统时间重新设置了随机数种子

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。
