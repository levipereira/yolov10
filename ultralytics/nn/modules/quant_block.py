
import torch.nn as nn

from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn.modules import _utils as quant_nn_utils
from pytorch_quantization.tensor_quant import QuantDescriptor

class QuantAdd(nn.Module, quant_nn_utils.QuantMixin):
    def __init__(self, quantization):
        super().__init__()
        
        if quantization:
            self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
            self._input1_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self.quantization = quantization

    def forward(self, x, y):
        if self.quantization:
            return self._input0_quantizer(x) + self._input1_quantizer(y)
        return x + y
