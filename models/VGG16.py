from torch import nn


def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU(inplace=True))
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*blk)


class VGG16CIFAR10(nn.Module):
    def __init__(self, conv_arch, fc_features, fc_hidden_units):
        super(VGG16CIFAR10, self).__init__()
        self.conv_arch = conv_arch
        self.fc_features = fc_features
        self.fc_hidden_units = fc_hidden_units
        self.conv_layer = nn.Sequential()
        for i, (num_convs, in_channels, out_channels) in enumerate(self.conv_arch):
            self.conv_layer.add_module('vgg_block_' + str(i + 1), vgg_block(num_convs, in_channels, out_channels))
        self.fc_layer = nn.Sequential(
            nn.Linear(self.fc_features, self.fc_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(self.fc_features, self.fc_hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(self.fc_hidden_units, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(-1, self.fc_features)
        x = self.fc_layer(x)
        return x


def form_model():
    conv_arch = ((2, 3, 64), (2, 64, 128), (3, 128, 256), (3, 256, 512), (3, 512, 512))
    fc_features = 512
    fc_hidden_units = 512

    model = VGG16CIFAR10(conv_arch, fc_features, fc_hidden_units)
    return model


if __name__ == '__main__':

    vgg16 = form_model()
    print(vgg16)
'''
    conv_layer.vgg_block_1
    .1.num_batches_tracked
    has
    a
    param: 79927
    conv_layer.vgg_block_1
    .4.running_var
    has
    a
    param: 13.907341957092285'''
