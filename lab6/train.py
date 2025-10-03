import torch
from torch import nn
from torchvision.transforms import ToTensor
from torchvision import datasets
from torch import optim
from torch.utils.data import DataLoader

class Perceptron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)

dataset_train = datasets.MNIST(
    root="MNIST",
    train=True,
    download=True,
    transform=ToTensor()
)

dataset_test = datasets.MNIST(
    root="MNIST",
    train=False,
    download=True,
    transform=ToTensor()
)

train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Perceptron(28*28, 28, 10).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(30):
    l = 0
    k = 0
    for input, target in train_loader:
        input, target = input.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(input.view(input.size(0), -1))
        loss = loss_fn(output, target)
        
        l += loss
        k += 1

        loss.backward()
        optimizer.step()

    print(l/k)

torch.save(model.state_dict(), "D:\Programming\python\Multimedia-Processing\lab6\models\model1.pth")