import torch
from torchvision.models import resnet

#model = resnet.resnet34(pretrained=True)
model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
model.eval()
traced_model = torch.jit.trace(model, torch.rand(1,3,224,224))
traced_model.save('mobilenetv2.pt')