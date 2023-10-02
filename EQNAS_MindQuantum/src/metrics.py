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
Custom binary cross entropy accuracy rate, used to output the Hamiltonian expected value range
[- 1,1] shift to [0,1] of qnn
"""
import mindspore.nn as nn
import numpy as np


class CustomizedBCEAcc(nn.Metric):
    def __init__(self):
        super(CustomizedBCEAcc, self).__init__()
        self.clear()

    def clear(self):
        self.sum = 0
        self.num = 0

    @nn.rearrange_inputs
    def update(self, *inputs):
        y_pred = ((inputs[0] + 1) / 2 >= 0.5).asnumpy().squeeze()  # ([-1,1] + 1)/2 = [0,1] >= 0.5 =[False, True]
        y_true = (inputs[1] >= 0.5).asnumpy()  # [0,1] >= 0.5 = [False, True]
        self.sum += np.sum((y_pred == y_true).astype(np.float32))  # [False, True] -> [0.,1.] ->sum = float_value
        self.num += y_true.shape[0]

    def eval(self):
        return self.sum / self.num
