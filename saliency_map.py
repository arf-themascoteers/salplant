import torch
from glasses_dataset import CustomImageDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def generate(device):
    torch.manual_seed(9)
    batch_size = 1
    cid = CustomImageDataset(is_train=False)
    dataloader = DataLoader(cid, batch_size=batch_size, shuffle=False)
    model = torch.load("cnn_trans.h5")
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    it = iter(dataloader)
    (x,y) = next(it)
    x = x.to(device)
    x.requires_grad_()
    y = y.to(device)
    print(y)
    y_hat = model(x)
    output_idx = y_hat.argmax()
    output_max = y_hat[0, output_idx]
    output_max.backward()
    saliency, _ = torch.max(x.grad.data.abs(), dim=1)
    saliency = saliency.reshape(224, 224)
    image = x.reshape(-1, 224, 224)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image.cpu().detach().numpy().transpose(1, 2, 0))
    ax[0].axis('off')
    ax[1].imshow(saliency.cpu(), cmap='hot')
    ax[1].axis('off')
    plt.tight_layout()
    fig.suptitle('The Image and Its Saliency Map')
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generate(device)
