import model
import utils
import argparse


import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from sklearn.model_selection import train_test_split



parser = argparse.ArgumentParser(prog="train.py", description="Train a model")
parser.add_argument("--data", type=str, help="Path to the data")
parser.add_argument("--model", type=str, default=None, help="Path to the model")
parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="beta 1")
parser.add_argument("--b2", type=float, default=0.999, help="beta 2")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")

args = parser.parse_args()


path_pairs = utils.get_file_paths(args.data + "hazy", args.data + "clear")


train_pairs, test_pairs = train_test_split(path_pairs, test_size=0.2, random_state=42)

train_dataset = utils.HeyZee(train_pairs)
test_dataset = utils.HeyZee(test_pairs)


train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

TheHayZee = model.MASK()

TheHayZee = TheHayZee.cuda()

loss = utils.TotalLoss()
optim = Adam(TheHayZee.parameters(), lr=args.lr, betas = (args.b1, args.b2), weight_decay=args.weight_decay)


for e in range(args.epochs):

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
