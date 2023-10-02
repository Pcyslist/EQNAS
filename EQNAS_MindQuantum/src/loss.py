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
Customized binary cross entropy loss, which is used to output the Hamiltonian expected value
range [- 1,1] shift to [0,1] of qnn.
"""
import mindspore.nn as nn
import mindspore as ms


class CustomizedBCELoss(nn.Cell):
    """Customized_BCELoss"""
    def __init__(self):
        super(CustomizedBCELoss, self).__init__()
        self.something = 1

    def construct(self, y_pred, y_true):
        loss = ms.nn.BCELoss(reduction='mean')
        y_pred = (y_pred + 1) / 2  # ([-1,1] + 1)/2 = [0,1]
        return loss(y_pred.squeeze(), y_true.astype(ms.float32))
