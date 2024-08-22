import modelopt.torch.quantization as mtq

config = mtq.INT8_DEFAULT_CFG


# Define forward loop for calibration
def forward_loop(model):
    for data in calib_set:
        model(data)


# QAT after replacement of regular modules to quantized modules
model = mtq.quantize(model, config, forward_loop)

# Fine-tune with original training pipeline
# Adjust learning rate and training duration
train(model, train_loader, optimizer, scheduler, ...)