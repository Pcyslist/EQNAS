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
"""Hyper-parameters."""

import numpy as np
from easydict import EasyDict

cfg = EasyDict()
cfg.LOG_NAME = "logger"

# Quantum evolution algorithm parameters
cfg.QEA = EasyDict()
cfg.QEA.fitness_best = []  # The best fitness of each generation

# Various parameters of the population
cfg.QEA.Genome = 64  # Chromosome length
cfg.QEA.N = 10  # Population size
cfg.QEA.generation_max = 10  # Population Iterations
cfg.QEA.av_fitness = []  # Average fitness of each generation
cfg.QEA.popSize = cfg.QEA.N + 1
cfg.QEA.genomeLength = cfg.QEA.Genome + 1
cfg.QEA.top_bottom = 3

# Parameters initialization
cfg.QEA.best_acc = np.empty([cfg.QEA.generation_max])  # Save the best accuracy of each generation
cfg.QEA.best_arch = np.empty([cfg.QEA.generation_max, cfg.QEA.genomeLength])
cfg.QEA.QuBitZero = np.array([[1], [0]])
cfg.QEA.QuBitOne = np.array([[0], [1]])
cfg.QEA.AlphaBeta = np.empty([cfg.QEA.top_bottom])
cfg.QEA.fitness = np.empty([cfg.QEA.popSize])

# Qpv: quantum chromosome (or population vector, QPV)
cfg.QEA.qpv = np.empty([cfg.QEA.popSize, cfg.QEA.genomeLength, cfg.QEA.top_bottom])
cfg.QEA.nqpv = np.empty([cfg.QEA.popSize, cfg.QEA.genomeLength, cfg.QEA.top_bottom])

# Chromosome: classical chromosome
cfg.QEA.chromosome = np.empty([cfg.QEA.popSize, cfg.QEA.genomeLength], dtype=np.int64)
cfg.QEA.child1 = np.empty([cfg.QEA.popSize, cfg.QEA.genomeLength, cfg.QEA.top_bottom])
cfg.QEA.child2 = np.empty([cfg.QEA.popSize, cfg.QEA.genomeLength, cfg.QEA.top_bottom])
cfg.QEA.best_chrom = np.empty([cfg.QEA.generation_max])

# Initialize QEA global variables
cfg.QEA.theta = 0
cfg.QEA.the_best_chrom = 0
cfg.QEA.generation = 0

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
