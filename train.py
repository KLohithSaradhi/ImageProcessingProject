import os
import utils
import model
from torch.utils.data import DataLoader
from torch.optim import Adam

import torch


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

path = "/Data"
path_pairs = utils.get_file_paths("./AugmentedData/hazy", "./AugmentedData/clear")


from sklearn.model_selection import train_test_split

train_pairs, test_pairs = train_test_split(path_pairs, test_size=0.2, random_state=42)

train_dataset = utils.HeyZee(train_pairs)
test_dataset = utils.HeyZee(test_pairs)

BATCH_SIZE = 1

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
TheHayZee = model.MASK()

TheHayZee = TheHayZee.cuda()

loss = utils.TotalLoss()
optim = Adam(TheHayZee.parameters(), lr=5e-4, betas = (0.5, 0.999), weight_decay=0.0001)
epochs = 35

for e in range(epochs):

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

    torch.save(TheHayZee.state_dict(), f"./weights/model_{e}.pth")