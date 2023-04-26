import torch
import torchvision

# Get pytorch model
base_path = "/home/lolo/omav_ws/src/rotors_simulator/rotors_description/models/"
model_path = base_path + "T_c.pt"
pytorch_model = torch.jit.load(model_path)

# Prepare stuff
dummy_input = torch.randn(8)

# Export
out_path = base_path + "T_c.onnx"
torch.onnx.export(pytorch_model, dummy_input, out_path, verbose=True)



