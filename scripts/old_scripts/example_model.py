import torch


model_dir = "../data/models/example_model/"
model_dir = "../data/models/saved/23-03-29--10-08-46_400Hz-L9-mixd-PHL08_Ta/"

model = torch.jit.load(model_dir + "delta_2000.pt")

input = torch.tensor([0,0,0,0,0,0,0.5,-1])

print(model(input))