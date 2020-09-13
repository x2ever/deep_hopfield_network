import torch


x = torch.randn((10, 16), requires_grad=True)
print(torch.mean(x))

for x_ in x:
    print(x_)

