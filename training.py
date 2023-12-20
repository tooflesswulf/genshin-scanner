import dataset
import torchvision.transforms.v2 as v2
import torchvision
import PIL

import torch
import numpy as np

from torch import nn
from torch.utils.data import DataLoader, RandomSampler

from pytorch_lightning import LightningModule, Trainer


class SimpleNetwork(nn.Module):
    def __init__(self, im_size=100, out_size=36):
        super(SimpleNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5),  # h-4, w-4
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # h/2, w/2
            nn.Conv2d(32, 64, 5),  # h-4, w-4
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # h/2, w/2
            nn.Conv2d(64, 128, 5),  # h-4, w-4
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # h/2, w/2
        )

        self.conv_size = im_size
        for _ in range(3):
            self.conv_size = (self.conv_size - 4) // 2
        self.fc_size = 128 * self.conv_size * self.conv_size
        self.fc = nn.Sequential(
            nn.Linear(self.fc_size, 64),
            nn.ReLU(),
            nn.Linear(64, out_size),
        )

    def forward(self, x, _=None):
        x = self.conv(x)
        x = x.view(-1, self.fc_size)
        x = self.fc(x)
        return x


class LModule(LightningModule):
    def __init__(self, model: nn.Module, lr=1e-5):
        super(LModule, self).__init__()
        self.model = model
        self.loss = nn.MSELoss()
        self.lr = lr

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return opt

    def training_step(self, batch, bi):
        x = batch['image']
        y = batch['y1']

        loss = self.loss(self.model(x), y)
        self.log('train_loss', loss, on_epoch=True, batch_size=x.shape[0])
        return loss

if __name__ == '__main__':
    tr = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.uint8, scale=True),
    v2.ColorJitter(brightness=.3, hue=.05, contrast=.1),
    v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Imagenet normalization
    ])

    im_size = 128

    ds = dataset.GenshinArtifactDataset('/home/code/scanner-data/', im_size=im_size, transform=tr)
    samp = RandomSampler(ds, replacement=True, num_samples=1024)
    loader = DataLoader(ds, batch_size=16, sampler=samp)

    model = SimpleNetwork(im_size=im_size)
    m = LModule(model, lr=1e-5)

    # Resume from:
    ckpt = torch.load('/home/code/lightning_logs/version_3/checkpoints/epoch=3001-step=96064.ckpt')
    m.load_state_dict(ckpt['state_dict'])

    trainer = Trainer(max_epochs=10_000, accelerator='gpu')
    trainer.fit(m, loader)

