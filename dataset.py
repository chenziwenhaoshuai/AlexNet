from torch.utils.data import Dataset
import cv2
import os
import torchvision.transforms as transforms
import torch



class AlexDataset(Dataset):
    def __init__(self):
        super(AlexDataset, self).__init__()
        self.datasets_path = './datasets/'
        self.dataset_list = os.listdir(self.datasets_path)

    def __getitem__(self, index):
        image_name = self.dataset_list[index]
        image_path = self.datasets_path + image_name
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        # plan 1
        # img = img/255.0
        # plan 2
        img = transforms.ToTensor()(img)
        label = image_name.split('_')[0]
        label = int(label)
        # label = self.one_hot(label, 2)
        return img,label

    def __len__(self):
        return len(self.dataset_list)

    def one_hot(self, label, num_class):
        one_hot = torch.zeros(num_class)
        one_hot[label] = 1
        return one_hot

if __name__ == '__main__':
    img, label = AlexDataset().__getitem__(100)
    print(img.shape)
    print(label.shape)
