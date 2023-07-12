import torch
import torch.nn.functional as F
from glasses_dataset import CustomImageDataset
from torch.utils.data import DataLoader
from traditional_machine import TraditionalMachine


def train(device):
    batch_size = 50
    cid = CustomImageDataset(is_train=True)
    dataloader = DataLoader(cid, batch_size=batch_size, shuffle=True)
    model = TraditionalMachine()
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    num_epochs = 3
    n_batches = int(len(cid)/batch_size) + 1
    batch_number = 0
    loss = None
    for epoch in range(num_epochs):
        batch_number = 0
        for (x, y) in dataloader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = F.cross_entropy(y_hat, y)
            loss.backward()
            optimizer.step()
            batch_number += 1
            print(f'Epoch:{epoch + 1} (of {num_epochs}), Batch: {batch_number} of ({n_batches}), Loss:{loss.item():.4f}')

    torch.save(model, 'cnn_trans.h5')


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(device)