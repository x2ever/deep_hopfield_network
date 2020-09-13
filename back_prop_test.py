import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc3 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.fc3(x)
        return x

# net = Net()
# x = torch.ones(2, requires_grad=True)
# y = 1
#
# optimizer = optim.Adam([x], lr=0.1)
# for _ in range(500):
#     optimizer.zero_grad()
#     loss = (net(x) - y) * (net(x) - y)
#     loss.backward()
#     optimizer.step()
#
#     print(loss, x, net(x))
from hopfield import HopfieldNetwork

hopfield_network = HopfieldNetwork(16)
x = torch.randn((10, 16), requires_grad=True)
hopfield_network.train(x)
print(hopfield_network.w)