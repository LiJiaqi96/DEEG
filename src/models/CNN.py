# coding:UTF-8
'''
对原始的 eeg 信号，使用 CNN 进行情感分类。
网络结构为：3层CNN+2层FC
'''

import torch
import torch.nn as nn




# 定义卷积网络结构
class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=(1, 1), padding=(2, 2), bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=(1, 1), padding=(2, 2), bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=(1, 1), padding=(1, 1), bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(128 * 7 * 25, 256, bias=True)
        self.fc2 = nn.Linear(256, num_classes, bias=True)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

#Example
'''
model = ConvNet(num_classes).to(device)

parameter:
num_classes:想要分成的情绪的种类


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.95)

'''
