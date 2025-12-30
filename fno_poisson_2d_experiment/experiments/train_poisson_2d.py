import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data.poisson_2d import create_dataset
from FNO.models.fno_2d import FNO2D


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    f, u = create_dataset(N=8, H=64, W=64)
    train_f, test_f = f[:6], f[6:]
    train_u, test_u = u[:6], u[6:]

    train_ds = TensorDataset(train_f, train_u)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)

    # Model
    model = FNO2D(
        in_channels=1,
        out_channels=1,
        modes1=16,
        modes2=16,
        hidden_channels=32,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Training
    for epoch in range(1,51):
        model.train()
        total_loss = 0.0

        for f_batch, u_batch in train_loader:
            f_batch = f_batch.to(device)
            u_batch = u_batch.to(device)

            optimizer.zero_grad()
            pred = model(f_batch)
            loss = loss_fn(pred, u_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch:03d} | Train Loss: {total_loss:.6f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        pred = model(test_f.to(device))
       
    print("Test completed.")
    return test_f, test_u, pred.cpu()


if __name__ == "__main__":
    train()
