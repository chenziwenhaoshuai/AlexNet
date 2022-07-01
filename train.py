import torch
import torch.nn as nn
from dataset import AlexDataset
from model import AlexNet
from tqdm import tqdm
from torch.utils.data import DataLoader


if __name__ == '__main__':
    lr = 0.001
    batch_size = 32
    num_epochs = 20
    num_class = 2
    dataset = AlexDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = AlexNet(num_class)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs): # epoch
        for i, (img, label) in tqdm(enumerate(dataloader)): # iter
            out = model(img)
            loss = loss_func(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch: {}, loss: {}'.format(epoch, loss.item()))
        torch.save(model.state_dict(), './model.pth')
    print('done')
