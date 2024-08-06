
import torch.nn as nn
import torch
import torch.nn.functional as F

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



class QuantADownAvgChunk(nn.Module):
    def __init__(self):
        super().__init__()
        self._chunk_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self._chunk_quantizer._calibrator._torch_hist = True
        self.avg_pool2d = nn.AvgPool2d(2, 1, 0, False, True)

    def forward(self, x):
        x = self.avg_pool2d(x)
        x = self._chunk_quantizer(x)
        return x.chunk(2, 1)

class QuantAConvAvgChunk(nn.Module):
    def __init__(self):
        super().__init__()
        self._chunk_quantizer = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8, calib_method="histogram"))
        self._chunk_quantizer._calibrator._torch_hist = True
        self.avg_pool2d = nn.AvgPool2d(2, 1, 0, False, True)

    def forward(self, x):
        x = self.avg_pool2d(x)
        x = self._chunk_quantizer(x)
        return x
    
class QuantRepNCSPELAN4Chunk(nn.Module):
    def __init__(self, c):
        super().__init__()
        self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self.c = c
    def forward(self, x, chunks, dims):
        return torch.split(self._input0_quantizer(x), (self.c, self.c), dims) 
       
class QuantUpsample(nn.Module): 
    def __init__(self, size, scale_factor, mode):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self._input_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        
    def forward(self, x):
        return F.interpolate(self._input_quantizer(x), self.size, self.scale_factor, self.mode) 

             
class QuantConcat(nn.Module): 
    def __init__(self, dim):
        super().__init__()
        self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self._input1_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self.dim = dim

    def forward(self, x, dim):
        x_0 = self._input0_quantizer(x[0])
        x_1 = self._input1_quantizer(x[1])
        return torch.cat((x_0, x_1), self.dim) 
    
class QuantC2fChunk(nn.Module):
    def __init__(self, c):
        super().__init__()
        self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self.c = c
    def forward(self, x, chunks, dims):
        return torch.split(self._input0_quantizer(x), (self.c, self.c), dims)
    