from models import UnetModel, UnetMOEModel
import torch

unet = UnetMOEModel(3, 1, 8)

out = unet(torch.rand(10, 3, 96, 96))

print(unet.state_dict)

print(unet.parameters_count())