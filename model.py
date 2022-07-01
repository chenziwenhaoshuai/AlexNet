import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self,num_class):
        super(AlexNet, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3,48,kernel_size=11,stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(48,128,kernel_size=5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(128,192,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,192,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,128,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.head = nn.Sequential(
            nn.Linear(128*5*5,2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048,2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(2048,num_class)
        )
        self.softmax = nn.Softmax()
    def forward(self, img):
        x = self.backbone(img)
        x = x.view(x.size(0),-1)
        x = self.head(x)
        # x = self.softmax(x)
        return x

if __name__ == '__main__':
    img = torch.randn(4,3,224,224) # N C H W
    model = AlexNet(num_class=2)
    out = model(img)

    # 2  [0.8,0.2]  3 [0.1,0.2,0.7] one hot