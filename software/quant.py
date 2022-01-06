import torch
from torch.quantization.fake_quantize import FakeQuantize
from torch.quantization.observer import MovingAveragePerChannelMinMaxObserver, MovingAverageMinMaxObserver
from torch.quantization import QConfig

def prepare_model(model):
    torch.backends.quantized.engine = 'fbgemm'
    model.fuse_model()
    model.qconfig = QConfig(activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                              quant_min=0,
                                                              quant_max=255,
                                                              reduce_range=True),
                            weight=FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver,
                                                          quant_min=-128,
                                                          quant_max=127,
                                                          dtype=torch.qint8,
                                                          qscheme=torch.per_channel_symmetric,
                                                          reduce_range=True,
                                                          ch_axis=0))
    torch.quantization.prepare_qat(model, inplace=True)
