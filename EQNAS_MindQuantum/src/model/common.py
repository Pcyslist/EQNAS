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
The common components of Quantum Neural Network Models.
"""
import numpy as np
import mindspore as ms
import mindquantum.core.gates as mq_gates
import mindquantum.core.circuit as mq_cir
import mindquantum.simulator as mq_sim
import mindquantum.core.operators as mq_ops
import mindquantum.framework as mq_frm
from src.utils.config import cfg


# Custom quantum gate matrix
def matrix(alpha):
    return np.array([[np.cos(np.pi * alpha / 2), np.sin(np.pi * alpha / 2)],
                     [np.sin(np.pi * alpha / 2), np.cos(np.pi * alpha / 2)]])


# Differential of custom quantum gate matrix
def diff_matrix(alpha):
    return 0.5 * np.pi * np.array([[-np.sin(np.pi * alpha / 2), np.cos(np.pi * alpha / 2)],
                                   [np.cos(np.pi * alpha / 2), -np.sin(np.pi * alpha / 2)]])


def create_qnn(chromosome_i):
    XI = mq_gates.gene_univ_parameterized_gate('XI', matrix, diff_matrix)
    # Construction of encoder quantum circuit
    encoder = mq_cir.Circuit()
    for i in range(1, 16 + 1):  # 17 qubits = 1 readout_qubit + 16 data_qubit(16 pixels)
        if cfg.DATASET.type == "mnist":
            encoder += XI(f'alpha{i}').on(i)
        else:
            encoder += mq_gates.RX(f'alpha{i}').on(i)
    # encoder does not require gradient, and the parameters in the line are provided by the features in the dataset
    encoder = encoder.no_grad()

    # Constructing ansatz quantum circuit
    ansatz = mq_cir.Circuit()
    ansatz += mq_gates.X.on(0)
    ansatz += mq_gates.H.on(0)
    for j in range(1, cfg.QEA.genomeLength, 2):
        if chromosome_i[j] == 0:
            if chromosome_i[j + 1] == 0:
                # xx
                ansatz += mq_gates.XX(f'XX_{(j // 2)}').on([0, (j // 2) % 16 + 1])
            else:
                # zz
                ansatz += mq_gates.ZZ(f'ZZ_{(j // 2)}').on([0, (j // 2) % 16 + 1])
        else:
            if chromosome_i[j + 1] == 0:
                # yy
                ansatz += mq_gates.YY(f'YY_{(j // 2)}').on([0, (j // 2) % 16 + 1])
            else:
                # I
                ansatz += mq_gates.I.on((j // 2) % 16 + 1)
    ansatz += mq_gates.H.on(0)
    # Constructing a Complete Quantum Neural Network
    circuit = encoder.as_encoder() + ansatz.as_ansatz()
    circuit.summary()
    # circuit.svg(style='light').to_file(filename=cfg.DATASET.type+"_qnn.svg")
    # Built on readout_ Hamiltonian measured in Z direction on qubit (q0)
    ham = mq_ops.Hamiltonian(mq_ops.QubitOperator('Z0'))
    # grad_ops
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="GPU")
    simulator = mq_sim.Simulator('projectq', circuit.n_qubits)
    grad_ops = simulator.get_expectation_with_grad(ham, circuit, parallel_worker=1)
    qnn = mq_frm.MQLayer(grad_ops)
    return qnn
