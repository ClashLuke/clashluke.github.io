import torch.nn as nn


class TinyCNN(nn.Module):
    def __init__(self, num_classes=10, width=64, dropout=0.0, use_bn=False):
        super().__init__()
        size, layers, in_ch, out_ch = 38, [], 3, width
        for i in range(3):
            layers.append(nn.Conv2d(in_ch, out_ch, 4, stride=2, bias=False))
            if i + 1 < 3:
                if use_bn:
                    layers.append(nn.BatchNorm2d(out_ch))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout2d(dropout))
            size = size // 2 - 1
            in_ch, out_ch = out_ch, out_ch * 2
        layers.append(nn.Flatten())
        inf = size ** 2 * in_ch
        if use_bn:
            layers += [nn.BatchNorm1d(inf), nn.ReLU()]
        else:
            layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(inf, 64, bias=False))
        if use_bn:
            layers += [nn.BatchNorm1d(64), nn.ReLU()]
        else:
            layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(64, num_classes, bias=False))
        self.net = nn.Sequential(*layers)

    @staticmethod
    def layer_names():
        return [f'conv{i}' for i in range(3)] + ['fc0', 'fc1']

    def forward(self, x):
        return self.net(x)
