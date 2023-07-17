from torch import nn


class VGG(nn.Module):

    def __init__(self, pretrained_model, freeze=True):
        super().__init__()

        self.model = pretrained_model
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.classifier[-1] = nn.Sequential(
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )
        # in_features = self.model._modules['classifier'][-1].in_features
        # out_features = 1
        # self.model._modules['classifier'][-1] = nn.Linear(in_features, out_features, bias=True)
        #TODO dimensions

    def forward(self, x):
        return self.model(x)
