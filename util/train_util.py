import torch

from torchvision.models import resnet18, resnet34, resnet50, \
                        ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
from torchvision.models import vgg11, vgg16, vgg19, \
                        VGG11_Weights, VGG16_Weights, VGG19_Weights
from torchvision.models import efficientnet_b3, efficientnet_b5, efficientnet_b7, \
                        EfficientNet_B3_Weights, EfficientNet_B5_Weights, EfficientNet_B7_Weights
from torchvision.models import densenet121, densenet169, densenet201, \
                        DenseNet121_Weights, DenseNet169_Weights, DenseNet201_Weights
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large, \
                        MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights

from model.models import *

def load_model(model_path, model, optimizer, device, device_ids):

    model_dict = torch.load(model_path)

    if not torch.cuda.is_available():
        device = "cpu"

    model = nn.DataParallel(model, device_ids=device_ids)
    model.module.load_state_dict(model_dict['model_state'])
    model = model.to(device)
    model_epoch = model_dict['epoch']

    if optimizer is not None:
        optimizer.load_state_dict(model_dict['optimizer_state'])

    return model, optimizer, model_epoch


def choose_model(name, freeze):
    if name == 'vgg11':
        return VGG(vgg11(weights=VGG11_Weights.DEFAULT), freeze=freeze)
    if name == 'vgg16':
        return VGG(vgg16(weights=VGG16_Weights.DEFAULT), freeze=freeze)
    if name == 'vgg19':
        return VGG(vgg19(weights=VGG19_Weights.DEFAULT), freeze=freeze)
    
    if name == 'resnet18':
        return ResNet(resnet18(weights=ResNet18_Weights.DEFAULT), freeze=freeze)
    if name == 'resnet34':
        return ResNet(resnet34(weights=ResNet34_Weights.DEFAULT), freeze=freeze)
    if name == 'resnet50':
        return ResNet(resnet50(weights=ResNet50_Weights.DEFAULT), freeze=freeze)
    
    if name == 'efficientnet_b3':
        return EfficientNet(efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT), freeze=freeze)
    if name == 'efficientnet_b5':
        return EfficientNet(efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT), freeze=freeze)
    if name == 'efficientnet_b7':
        return EfficientNet(efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT), freeze=freeze)
    
    if name == 'densenet121':
        return DenseNet(densenet121(weights=DenseNet121_Weights.DEFAULT), freeze=freeze)
    if name == 'densenet169':
        return DenseNet(densenet169(weights=DenseNet169_Weights.DEFAULT), freeze=freeze)
    if name == 'densenet201':
        return DenseNet(densenet201(weights=DenseNet201_Weights.DEFAULT), freeze=freeze)
    
    if name == 'mobilenetv3_small':
        return MobileNet(
            mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT), freeze=freeze)
    if name == 'mobilenetv3_large':
        return MobileNet(
            mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT), freeze=freeze)