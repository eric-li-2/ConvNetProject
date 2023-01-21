import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import optim
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(start_dim=1),
            nn.Linear(800, 400),
            nn.ReLU(),
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        logits = self.conv(x)
        return logits


def get_data(batch_size):
    training_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    # print(training_data[0][0].shape)
    # tensor_image = training_data[0][0].permute(1,2,0)
    # plt.imshow(tensor_image, cmap="gray")
    # plt.show()
    # labels_map = {
    # 0: "airplane",
    # 1: "automobile",
    # 2: "bird",
    # 3: "cat",
    # 4: "deer",
    # 5: "dog",
    # 6: "frog",
    # 7: "horse",
    # 8: "ship",
    # 9: "truck",
    # }
    # figure = plt.figure(figsize=(8, 8))
    # cols, rows = 3, 3
    # for i in range(1, cols * rows + 1):
    #     sample_idx = torch.randint(len(training_data), size=(1,)).item()
    #     img, label = training_data[sample_idx]
    #     figure.add_subplot(rows, cols, i)
    #     plt.title(labels_map[label])
    #     plt.axis("off")
    #     plt.imshow(img.permute(1,2,0), cmap="gray")
    # plt.show()
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    return train_dataloader, test_dataloader


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def fit(epochs, model, loss_fn, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_fn, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_fn, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)

def main():
    epochs = 20
    batch_size = 64
    lr = 0.1
    momentum = 0

    train_dl, test_dl = get_data(batch_size)
    model = NeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    # fit(epochs, model, loss_fn, opt, train_dl, test_dl)
    for t in range(epochs):
        print(f"Epoch {t+1}\n------------------------------")
        train(train_dl, model, loss_fn, opt)
        test(test_dl, model, loss_fn)
    
    print("Done!")

main()
