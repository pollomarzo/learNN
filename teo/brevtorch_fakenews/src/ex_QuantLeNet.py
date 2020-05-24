"""
quantlenet implemented as brevitas example
"""
from torch.nn import Module
import torch.nn.functional as F
import brevitas.nn as qnn
from brevitas.core.quant import QuantType


class QuantLeNet(Module):
    def __init__(self):
        super(QuantLeNet, self).__init__()
        self.conv1 = qnn.QuantConv2d(3, 6, 5,
                                     weight_quant_type=QuantType.INT,
                                     weight_bit_width=8)
        self.relu1 = qnn.QuantReLU(
            quant_type=QuantType.INT, bit_width=8, max_val=6)
        self.conv2 = qnn.QuantConv2d(6, 16, 5,
                                     weight_quant_type=QuantType.INT,
                                     weight_bit_width=8)
        self.relu2 = qnn.QuantReLU(
            quant_type=QuantType.INT, bit_width=8, max_val=6)
        self.fc1 = qnn.QuantLinear(16*5*5, 120, bias=True,
                                   weight_quant_type=QuantType.INT,
                                   weight_bit_width=8)
        self.relu3 = qnn.QuantReLU(
            quant_type=QuantType.INT, bit_width=8, max_val=6)
        self.fc2 = qnn.QuantLinear(120, 84, bias=True,
                                   weight_quant_type=QuantType.INT,
                                   weight_bit_width=8)
        self.relu4 = qnn.QuantReLU(
            quant_type=QuantType.INT, bit_width=8, max_val=6)
        self.fc3 = qnn.QuantLinear(84, 10, bias=False,
                                   weight_quant_type=QuantType.INT,
                                   weight_bit_width=8)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = self.relu2(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.relu3(self.fc1(out))
        out = self.relu4(self.fc2(out))
        out = self.fc3(out)
        return out
