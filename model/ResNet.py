from torch import nn


class ResNet(nn.Module):

    def __init__(self, pretrained_model, freeze=True):
        super().__init__()

        self.model = pretrained_model
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid()
        )
        #TODO dimensions

    def forward(self, x):
        return self.model(x)
