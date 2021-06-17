import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers.classifier import ClassifierHead
import torchvision
import functools
import operator
from efficientnet_pytorch import EfficientNet

class Network(nn.Module): # pretrained model classifier를 num_classes에 맞추어 변형 해주는 class
    def __init__(self, pretrained_model, out_features, input_dim=(3, 224, 224)):
        super(Network, self).__init__()
        # Load pretrained model, only feature extractors
        self.backbone = nn.Sequential(*(list(pretrained_model.children())[:-1]))
        # Auto-calculate input for the fc layers
        num_features_before_fcnn = functools.reduce(operator.mul,
                                                    list(self.backbone(torch.rand(1, *input_dim)).shape))
        # Fc layer
        self.fc1 = nn.Sequential(nn.Linear(in_features=num_features_before_fcnn, out_features=out_features), )

    def forward(self, x):
        output = self.backbone(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output



class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU())


# Residual block
class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()

        reduced_channels = int(in_channels/2)

        self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=1, padding=0)
        self.layer2 = conv_batch(reduced_channels, in_channels)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out


class Darknet53(nn.Module):
    def __init__(self, block, num_classes):
        super(Darknet53, self).__init__()

        self.num_classes = num_classes
        self.conv1 = conv_batch(3, 32)
        self.conv2 = conv_batch(32, 64, stride=2)
        self.residual_block1 = self.make_layer(block, in_channels=64, num_blocks=1)
        self.conv3 = conv_batch(64, 128, stride=2)
        self.residual_block2 = self.make_layer(block, in_channels=128, num_blocks=2)
        self.conv4 = conv_batch(128, 256, stride=2)
        self.residual_block3 = self.make_layer(block, in_channels=256, num_blocks=8)
        self.conv5 = conv_batch(256, 512, stride=2)
        self.residual_block4 = self.make_layer(block, in_channels=512, num_blocks=8)
        self.conv6 = conv_batch(512, 1024, stride=2)
        self.residual_block5 = self.make_layer(block, in_channels=1024, num_blocks=4)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mask_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3),
        )
        self.gender_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
        )
        self.age_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.residual_block1(out)
        out = self.conv3(out)
        out = self.residual_block2(out)
        out = self.conv4(out)
        out = self.residual_block3(out)
        out = self.conv5(out)
        out = self.residual_block4(out)
        out = self.conv6(out)
        out = self.residual_block5(out)
        out = self.global_avg_pool(out)
        out = out.view(-1, 1024)
        age = self.age_classifier(out)
        gender = self.gender_classifier(out)
        out = self.mask_classifier(out) # mask인데 메모리 절약차원에서..

        return out, gender, age

    def make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)


def darknet53(num_classes):
    return Darknet53(DarkResidualBlock, num_classes)

class resnet50(nn.Module): # 18 class 분류
    def __init__(self, num_classes):
        super(resnet50, self).__init__()

        self.net = torchvision.models.resnet.resnet50()
        self.net.fc = nn.Linear(in_features=2048, out_features=18)


    def forward(self, x):
        return self.net(x)

class custom_resnet50(nn.Module): # mask, age, gender 클래스 별 분류
    def __init__(self, num_classes):
        super(custom_resnet50, self).__init__()

        model = timm.create_model('resnet50', pretrained=True)
        self.backbone = nn.Sequential(*(list(model.children())[:-2]))
        self.mask_classifier = ClassifierHead(2048, 3)
        self.gender_classifier = ClassifierHead(2048, 2)
        self.age_classifier = ClassifierHead(2048, 3)


    def forward(self, x):
        x = self.backbone(x)
        z = self.age_classifier(x)
        y = self.gender_classifier(x)
        x = self.mask_classifier(x)
        return x, y, z

class efficientnet_b3(nn.Module): # 18 class 분류
    def __init__(self, num_classes):
        super(efficientnet_b3, self).__init__()

        model = timm.create_model('efficientnet_b3', pretrained=True)
        self.backbone = nn.Sequential(*(list(model.children())[:-2]))

        self.classifier = ClassifierHead(1536, 18)


    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

class custom_efficientnet_b3(nn.Module): # mask, age, gender 클래스 별 분류
    def __init__(self, num_classes):
        super(custom_efficientnet_b3, self).__init__()

        model = timm.create_model('efficientnet_b3', pretrained=True)
        self.backbone = nn.Sequential(*(list(model.children())[:-2]))

        self.age_classifier = ClassifierHead(1536, 3)
        self.gender_classifier = ClassifierHead(1536, 2)
        self.mask_classifier = ClassifierHead(1536, 3)

    def forward(self, x):
        x = self.backbone(x)
        z = self.age_classifier(x)
        y = self.gender_classifier(x)
        x = self.mask_classifier(x)
        return x, y, z

