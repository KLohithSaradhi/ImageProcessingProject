import utils
import model
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from sklearn.model_selection import train_test_split

PATH_PAIRS = utils.get_file_paths("./AugmentedData/hazy/", "./AugmentedData/clear/")

train_paths, test_paths = train_test_split(PATH_PAIRS, test_size=0.2)

train_dataset = utils.HeyZee(train_paths)
test_dataset = utils.HeyZee(test_paths)

BATCH_SIZE = 4

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

TheHayZee = model.MASK()
TheHayZee = TheHayZee.cuda()

loss = utils.TotalLoss()
optim = Adam(TheHayZee.parameters(), lr=5e-4, betas = (0.5, 0.999), weight_decay=0.0001)

EPOCHS = 10


for e in range(EPOCHS):

    for i, (haze, clear) in enumerate(train_loader):

        I0, I1, I2, I3 = haze
        O0, O1, O2, O3 = clear

        I0 = I0.cuda()
        I1 = I1.cuda()
        I2 = I2.cuda()
        I3 = I3.cuda()

        O0 = O0.cuda()
        O1 = O1.cuda()
        O2 = O2.cuda()
        O3 = O3.cuda()
        
        optim.zero_grad()
        out = TheHayZee(I0, I1, I2, I3)

        loss_val = loss((O0, O1, O2, O3), out)

        loss_val.backward()
        optim.step()

        print(f"Epoch {e} Iteration {i} Loss {loss_val.item()}")

    torch.save(TheHayZee.state_dict(), f"./model_{e}.pth")
