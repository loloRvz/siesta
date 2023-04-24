import torch
import torchvision

# Get pytorch model
model_path = "../data/models/saved/23-03-29--10-08-46_400Hz-L9-mixd-PHL08_Ta/delta_2000.pt"
pytorch_model = torch.jit.load(model_path)

# Prepare stuff
dummy_input = torch.randn(8)

out_path = "../data/T_a.onnx"

torch.onnx.export(pytorch_model, dummy_input, out_path, verbose=True)



