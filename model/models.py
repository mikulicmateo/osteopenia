from torch import nn


class ResNet(nn.Module):

    def __init__(self, pretrained_model, freeze=True):
        super().__init__()

        self.model = pretrained_model
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        in_features=self.model.fc.in_features
        out_features=1

        self.model.fc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class VGG(nn.Module):

    def __init__(self, pretrained_model, freeze=True):
        super().__init__()

        self.model = pretrained_model
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        in_features = self.model._modules['classifier'][-1].in_features
        out_features = 1

        self.model.classifier[-1] = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class EfficientNet(nn.Module):

    def __init__(self, pretrained_model, freeze=True):
        super().__init__()

        self.model = pretrained_model
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        in_features = self.model._modules['classifier'][-1].in_features
        out_features = 1

        self.model.classifier[-1] = nn.Sequential(
            nn.Dropout(p = 0.2, inplace=True),
            nn.Linear(in_features, out_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    

class DenseNet(nn.Module):

    def __init__(self, pretrained_model, freeze=True):
        super().__init__()

        self.model = pretrained_model
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        in_features = self.model._modules['classifier'].in_features
        out_features = 1

        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    

class MobileNet(nn.Module):

    def __init__(self, pretrained_model, freeze=True):
        super().__init__()

        self.model = pretrained_model
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        in_features = self.model._modules['classifier'][-1].in_features
        out_features = 1

        self.model.classifier[-1] = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)