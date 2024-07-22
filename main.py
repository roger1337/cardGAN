
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn


CHANNELS = 3
OPTIMISER_LR = 3e-4
Z_DIM = 100
IMAGE_DIM = 128*128
X_DIM = 128
Y_DIM = 128
BATCH_SIZE = 5  
EPOCHS = 100  
DATA_PATH = "./images"
TRANSFORM = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize( (128, 128) )
        ]
)
DEVICE = "cuda"
LOG_STEP = 20

class Discriminator(nn.Module):
    def __init__(self, inFeatures, hiddenDim=2048, lr=0.01):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(CHANNELS, 32, 4, stride=2, padding=1, bias=False), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32 * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32 * 2, 32 * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32 * 4, 32 * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32 * 8, 32 * 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32 * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32 * 16, 1, 4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

class Generator(nn.Module):
    def __init__(self, zDim, imgDim, hiddenDim=2048, lr=0.01):
        super().__init__()
        self.fc = nn.Sequential(
            nn.ConvTranspose2d(Z_DIM, 32 * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(32 * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(32 * 16, 32 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32 * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(32 * 8, 32 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32 * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(32 * 4, 32 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32 * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(32 * 2,     32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.fc(x)



dataset = torchvision.datasets.ImageFolder(root=DATA_PATH, transform=TRANSFORM) 
loader = DataLoader(dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=True)
discriminator = Discriminator(IMAGE_DIM).to(DEVICE)
generator = Generator(Z_DIM, IMAGE_DIM).to(DEVICE)
discriminatorOptim = torch.optim.Adam(discriminator.parameters(), lr=OPTIMISER_LR)
generatorOptim = torch.optim.Adam(generator.parameters(), lr=OPTIMISER_LR)
loss = nn.BCELoss()
noise = torch.randn(BATCH_SIZE,  Z_DIM).to(DEVICE)
writerFake = SummaryWriter(f"logs/fake")
writerReal = SummaryWriter(f"logs/real")

def prepareVisualization(epoch,
                         batchIdx,
                         loaderLen,
                         lossD,
                         lossG,
                         writerFake,
                         writerReal,
                         step):
    print(
        f"Epoch [{epoch}/{EPOCHS}] Batch {batchIdx}/{loaderLen} \
                              Loss DISC: {lossD:.4f}, loss GEN: {lossG:.4f}"
    )

    with torch.no_grad():
        fake = generator.fc(noise).reshape(-1, 3, X_DIM, Y_DIM)
        real = real.reshape(-1, 3, X_DIM, Y_DIM)
        imgGridFake = torchvision.utils.make_grid(fake, normalize=True)
        imgGridReal = torchvision.utils.make_grid(real, normalize=True)
        writerFake.add_image("fake",imgGridFake, global_step=step)
        writerReal.add_image("real", imgGridReal, global_step=step)
        step += 1

    return step

step = 0
print(f"\nStarted Training and visualization...")
for epoch in range(EPOCHS):
    print('-' * 80)
    for batch_idx, (real, _) in enumerate(loader):
        realImage = real.to(DEVICE)
        noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(DEVICE)
        fakeImage = generator.fc(noise)


        discriminatorReal = discriminator.fc(realImage).view(-1)
        lossReal = loss(discriminatorReal, torch.ones_like(discriminatorReal))
        discriminatorFake = discriminator.fc(fakeImage).view(-1)
        lossFake = loss(discriminatorFake, torch.zeros_like(discriminatorFake))
        lossD = (lossReal + lossFake) / 2

        discriminator.fc.zero_grad()
        lossD.backward(retain_graph=True)
        discriminatorOptim.step()

        ###
        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # where the second option of maximizing doesn't suffer from
        # saturating gradients. Minimizing is easier
        ###
        output = discriminator.fc(fakeImage).view(-1)
        lossG = loss(output, torch.ones_like(output))
        generator.fc.zero_grad()
        lossG.backward()
        generatorOptim.step()

              # Visualize three steps for each epoch
        if batch_idx % LOG_STEP == 0:
            step = prepareVisualization(epoch,
                                        batch_idx,
                                        len(loader),
                                        lossD,
                                        lossG,
                                        writerFake,
                                        writerReal,
                                        step)



