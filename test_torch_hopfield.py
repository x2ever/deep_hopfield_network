
import torch
import torch.nn.functional as F


class Encoder(torch.nn.Module):
    def __init__(self, latent_size):
        super(Encoder, self).__init__()
        self.latent_size = latent_size

        self.conv1 = torch.nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = torch.nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = torch.nn.Linear(64 * 7 * 7, latent_size)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 64 * 7 * 7)   # reshape Variable
        x = self.fc1(x)
        return x


class FCN(torch.nn.Module):
    def __init__(self, label_size, latent_size):
        super(FCN, self).__init__()
        self.latent_size = latent_size
        self.label_size = label_size
        self.l1 = torch.nn.Linear(self.latent_size, self.label_size)

    def forward(self, x):

        return F.softmax(self.l1(x), dim=1)


class HopfieldNet(torch.nn.Module):
    def __init__(self, label_size, latent_size):
        super(HopfieldNet, self).__init__()
        self.latent_size = latent_size
        self.label_size = label_size
        self.w = torch.zeros((self.latent_size, self.latent_size))

    def forward(self, x):
        rho = torch.mean(x)

        for i, x_ in enumerate(x):
            temp = x_ - rho
            if i == 0:
                self.w = torch.ger(temp, temp)
            else:
                self.w += torch.ger(temp, temp)

        diag_w = torch.diag(torch.diag(self.w))
        self.w = self.w - diag_w
        self.w /= self.label_size

        return self.w


class DeepHopfield(torch.nn.Module):
    def __init__(self, input_size, label_size, latent_size):
        super(DeepHopfield, self).__init__()
        self.label_size = label_size
        self.latent_size = latent_size
        self.input_size = input_size
        self.label_images = torch.rand((self.label_size, 1, 28, 28), requires_grad=False).cuda()
        self.encoder = Encoder(latent_size).to(torch.device("cuda"))
        self.hopfield_net = HopfieldNet(label_size, latent_size)
        self.fully_connected_network = FCN(label_size, latent_size)

    def forward(self, image):
        represent_latent_vectors = torch.tanh(self.encoder(self.label_images))
        represent_latent_vectors = represent_latent_vectors.to(torch.device("cpu"))
        latent_vectors = self.encoder(image)
        latent_vectors = latent_vectors.to(torch.device("cpu"))

        label = self.fully_connected_network(latent_vectors.clone().detach())
        hopfield_weight = self.hopfield_net(represent_latent_vectors)

        for i, latent_vector in enumerate(latent_vectors):
            latent_vectors[i] = self.clustering(latent_vector, hopfield_weight)

        return F.softmax(torch.abs(latent_vectors @ represent_latent_vectors.T), dim=1), label

    def clustering(self, init_s, w):
        # Compute initial state energy
        s = torch.tanh(init_s)

        min_e = float('inf')
        min_s = None
        for _ in range(self.latent_size):
            prev_s = s.clone().detach()
            s = torch.abs(prev_s) * torch.sign(w @ prev_s)

            e = self.energy(s, w)

            if min_e > e:
                min_e = e
                min_s = s

        return min_s

    @staticmethod
    def energy(s, w):
        return - s @ w @ s

    def optimzer(self):
        opt1 = torch.optim.Adam([
                {'params': self.encoder.parameters()},
                {'params': self.fully_connected_network.parameters()}
            ], lr=0.001)
        opt2 = torch.optim.Adam([self.label_images], lr=0.001)

        return opt1, opt2


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from PIL import Image
    import numpy as np
    import os

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()), batch_size=512, shuffle=True
    )

    model = DeepHopfield(784, 10, 64)
    opt1, opt2 = model.optimzer()

    for step in range(10000):
        for b, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            opt1.zero_grad()
            opt2.zero_grad()

            clustered_label, label = model(data)
            klv_loss = torch.nn.KLDivLoss(reduction='batchmean')
            loss = klv_loss(clustered_label, label)
            print(f"[Epoch: {step: 05d} | {b: 02d} / {len(train_loader) - 1: 02d}] loss: {loss.data: .5f}", end='\r')
            loss.backward()
            opt1.step()
            opt2.step()

            model.label_images = torch.nn.Parameter(
                (model.label_images - torch.min(model.label_images)) /
                (torch.max(model.label_images) - torch.min(model.label_images))
            )
        print("")
        if step % 10 == 0:
            images = 255 * model.label_images.detach().cpu().numpy()
            os.makedirs(f'{step: 05d}', exist_ok=True)
            for i, image in enumerate(images):
                image = np.reshape(image, (28, 28))
                img = Image.fromarray(image)
                img = img.convert('L')
                img.save(f'{step: 05d}/{i}.jpg')

