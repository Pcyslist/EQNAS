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
The implementation of quantum evolutionary algorithm
"""
import logging
import math
import numpy as np
from src.utils.config import cfg
from src.utils.train_utils import train_qnn


def init_population():
    # Hadamard gate
    r2 = math.sqrt(2.0)
    h = np.array([[1 / r2, 1 / r2], [1 / r2, -1 / r2]])
    # Rotation Q-gate
    theta = 0
    rot = np.empty([2, 2])
    # Initial population array (individual x chromosome)
    i = 1
    j = 1
    for i in range(1, cfg.QEA.popSize):
        for j in range(1, cfg.QEA.genomeLength):
            theta = np.random.uniform(0, 1) * 90
            theta = math.radians(theta)
            rot[0, 0] = math.cos(theta)
            rot[0, 1] = -math.sin(theta)
            rot[1, 0] = math.sin(theta)
            rot[1, 1] = math.cos(theta)
            cfg.QEA.AlphaBeta[0] = rot[0, 0] * (h[0][0] * cfg.QEA.QuBitZero[0]) + rot[0, 1] * (h[0][1] *
                                                                                               cfg.QEA.QuBitZero[1])
            cfg.QEA.AlphaBeta[1] = rot[1, 0] * (h[1][0] * cfg.QEA.QuBitZero[0]) + rot[1, 1] * (h[1][1] *
                                                                                               cfg.QEA.QuBitZero[1])
            # alpha squared
            cfg.QEA.qpv[i, j, 0] = np.around(2 * pow(cfg.QEA.AlphaBeta[0], 2), 2)
            # beta squared
            cfg.QEA.qpv[i, j, 1] = np.around(2 * pow(cfg.QEA.AlphaBeta[1], 2), 2)


# Obverse
def measure(p_alpha):
    for i in range(1, cfg.QEA.popSize):
        for j in range(1, cfg.QEA.genomeLength):
            if p_alpha <= cfg.QEA.qpv[i, j, 0]:
                cfg.QEA.chromosome[i, j] = 0
            else:
                cfg.QEA.chromosome[i, j] = 1


def fitness_evaluation(generation):
    # i = 1
    # j = 1
    fitness_total = 0
    sum_sqr = 0
    # fitness_average = 0
    # variance = 0
    for i in range(1, cfg.QEA.popSize):
        cfg.QEA.fitness[i] = 0
        # Constructing quantum neural network based on chromosome
        qnn_results = train_qnn(cfg.QEA.chromosome[i], generation, i, cfg.TRAIN.EPOCHS)
        cfg.QEA.fitness[i] = qnn_results
        # logger.info("fitness = ",f," ",fitness[i])
        fitness_total = fitness_total + cfg.QEA.fitness[i]
    fitness_average = fitness_total / cfg.QEA.N
    cfg.QEA.av_fitness.append(fitness_average)
    i = 1
    while i <= cfg.QEA.N:
        sum_sqr = sum_sqr + pow(cfg.QEA.fitness[i] - fitness_average, 2)
        i = i + 1
    variance = sum_sqr / cfg.QEA.N
    if variance <= 1.0e-4:
        variance = 0.0
    # Best chromosome selection
    the_best_chrom = 1
    fitness_max = cfg.QEA.fitness[1]
    for i in range(1, cfg.QEA.popSize):
        if cfg.QEA.fitness[i] >= fitness_max:
            fitness_max = cfg.QEA.fitness[i]
            the_best_chrom = i
    cfg.QEA.best_chrom[generation] = the_best_chrom
    cfg.QEA.fitness_best.append(cfg.QEA.fitness[the_best_chrom])
    cfg.QEA.best_acc[generation] = cfg.QEA.fitness[the_best_chrom]
    cfg.QEA.best_arch[generation] = cfg.QEA.chromosome[the_best_chrom]
    logger = logging.getLogger(cfg.LOG_NAME)
    logger.info("Population size = %s", str(cfg.QEA.popSize - 1))
    logger.info("mean fitness = %s", str(fitness_average))
    lg_info = "variance = " + str(variance) + " Std. deviation = " + str(math.sqrt(variance))
    logger.info(lg_info)
    logger.info("fitness max = %s", str(cfg.QEA.fitness[the_best_chrom]))


# Update by quantum rotation gate
def rotation(generation):
    rot = np.empty([2, 2])
    # Lookup table of the rotation angle
    for i in range(1, cfg.QEA.popSize):
        for j in range(1, cfg.QEA.genomeLength):
            if cfg.QEA.fitness[i] < cfg.QEA.fitness[int(cfg.QEA.best_chrom[generation])]:
                if cfg.QEA.chromosome[i, j] == 0 and cfg.QEA.chromosome[int(cfg.QEA.best_chrom[generation]), j] == 1:
                    # rotating angle 0.03pi
                    delta_theta = 0.0932477796
                    rot[0, 0] = math.cos(delta_theta)
                    rot[0, 1] = -math.sin(delta_theta)
                    rot[1, 0] = math.sin(delta_theta)
                    rot[1, 1] = math.cos(delta_theta)
                    cfg.QEA.nqpv[i, j, 0] = (rot[0, 0] * cfg.QEA.qpv[i, j, 0]) + (rot[0, 1] * cfg.QEA.qpv[i, j, 1])
                    cfg.QEA.nqpv[i, j, 1] = (rot[1, 0] * cfg.QEA.qpv[i, j, 0]) + (rot[1, 1] * cfg.QEA.qpv[i, j, 1])
                    cfg.QEA.qpv[i, j, 0] = round(cfg.QEA.nqpv[i, j, 0], 2)
                    cfg.QEA.qpv[i, j, 1] = round(1 - cfg.QEA.nqpv[i, j, 0], 2)
                if cfg.QEA.chromosome[i, j] == 1 and cfg.QEA.chromosome[int(cfg.QEA.best_chrom[generation]), j] == 0:
                    delta_theta = -0.0942477796
                    rot[0, 0] = math.cos(delta_theta)
                    rot[0, 1] = -math.sin(delta_theta)
                    rot[1, 0] = math.sin(delta_theta)
                    rot[1, 1] = math.cos(delta_theta)
                    cfg.QEA.nqpv[i, j, 0] = (rot[0, 0] * cfg.QEA.qpv[i, j, 0]) + (rot[0, 1] * cfg.QEA.qpv[i, j, 1])
                    cfg.QEA.nqpv[i, j, 1] = (rot[1, 0] * cfg.QEA.qpv[i, j, 0]) + (rot[1, 1] * cfg.QEA.qpv[i, j, 1])
                    cfg.QEA.qpv[i, j, 0] = round(cfg.QEA.nqpv[i, j, 0], 2)
                    cfg.QEA.qpv[i, j, 1] = round(1 - cfg.QEA.nqpv[i, j, 0], 2)


# Quantum chromosome mutation
def mutation(pop_mutation_rate, mutation_rate):
    for i in range(1, cfg.QEA.popSize):
        up = np.random.random_integers(100)
        up = up / 100
        if up <= pop_mutation_rate:
            for j in range(1, cfg.QEA.genomeLength):
                um = np.random.random_integers(100)
                um = um / 100
                if um <= mutation_rate:
                    cfg.QEA.nqpv[i, j, 0] = cfg.QEA.qpv[i, j, 1]
                    cfg.QEA.nqpv[i, j, 1] = cfg.QEA.qpv[i, j, 0]
                else:
                    cfg.QEA.nqpv[i, j, 0] = cfg.QEA.qpv[i, j, 0]
                    cfg.QEA.nqpv[i, j, 1] = cfg.QEA.qpv[i, j, 1]
        else:
            for j in range(1, cfg.QEA.genomeLength):
                cfg.QEA.nqpv[i, j, 0] = cfg.QEA.qpv[i, j, 0]
                cfg.QEA.nqpv[i, j, 1] = cfg.QEA.qpv[i, j, 1]
    for i in range(1, cfg.QEA.popSize):
        for j in range(1, cfg.QEA.genomeLength):
            cfg.QEA.qpv[i, j, 0] = cfg.QEA.nqpv[i, j, 0]
            cfg.QEA.qpv[i, j, 1] = cfg.QEA.nqpv[i, j, 1]


# Entirety interference crossover
def all_cross():
    qpv_new = cfg.QEA.qpv
    qpv_a = cfg.QEA.qpv
    qpv_swap = cfg.QEA.qpv[1, 1]
    logger = logging.getLogger(cfg.LOG_NAME)
    logger.info(cfg.QEA.qpv[2, 2])
    for i in range(1, cfg.QEA.popSize):
        for j in range(1, cfg.QEA.genomeLength):
            if j >= 2:
                k = i - j + 1
                if i - j + 1 <= 0:
                    k = k - 1
                qpv_swap = qpv_a[i, k]
                qpv_new[i, j] = qpv_swap
                cfg.QEA.qpv[i, j] = qpv_new[i, j]
    logger.info(cfg.QEA.qpv[2, 2])


# qea
def qea():
    generation = 0
    logger = logging.getLogger(cfg.LOG_NAME)
    lg_info = "============== GENERATION: " + str(generation) + " =========================== \n"
    logger.info(lg_info)
    init_population()
    # Show_population()
    while generation < cfg.QEA.generation_max:
        measure(0.5)
        fitness_evaluation(generation)
        lg_info = "The best arch index of generation [" + str(generation) + "] is " + \
                  str(cfg.QEA.best_chrom[generation]) + '\n'
        logger.info(lg_info)
        rotation(generation)
        mutation(0.01, 0.002)
        # all_cross()
        generation = generation + 1
        lg_info = "============== GENERATION: " + str(generation) + " =========================== \n"
        logger.info()
