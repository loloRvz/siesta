import torch
import torchvision.models as models

vgg16 = models.vgg16(pretrained=False)

vgg16.eval()

in_tensor = torch.rand(size=(10, 3, 224, 224))
out_tensor = vgg16.forward(in_tensor)
print(out_tensor)

vgg16_traced = torch.jit.trace(vgg16, in_tensor)

base_path = "/home/lolo/omav_ws/src/rotors_simulator/rotors_description/models/"
out_path = base_path + "vgg16.pt"
vgg16_traced.save(out_path)
