import torch
import torch.nn as nn
from dataset import AlexDataset
from model import AlexNet
from tqdm import tqdm
from torchvision.transforms import transforms
import cv2
from torch.functional import F
# from torch.utils.data import DataLoader


if __name__ == '__main__':
    image_path = './test/1_000.jpg'
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    model = AlexNet(2)
    model.load_state_dict(torch.load('./model.pth'))
    out = model(img)
    out = F.softmax(out, dim=1)
    print(out)
