
import torch
import torch.nn.functional as F

device = torch.device("cuda")


class Softmax(torch.nn.Module):
    def __init__(self, latent_size, label_size):
        super(Softmax, self).__init__()

        self.fc1 = torch.nn.Linear(latent_size, label_size)

    def forward(self, x):
        return F.softmax(self.fc1(x), dim=1)


class Encoder(torch.nn.Module):
    def __init__(self, latent_size):
        super(Encoder, self).__init__()
        self.latent_size = latent_size

        self.conv1 = torch.nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = torch.nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = torch.nn.Linear(64 * 28 * 28, latent_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64 * 28 * 28)   # reshape Variable
        x = self.fc1(x)
        return torch.tanh(x)


class Decoder(torch.nn.Module):
    def __init__(self, latent_size):
        super(Decoder, self).__init__()
        self.latent_size = latent_size

        self.conv1 = torch.nn.ConvTranspose2d(32, 1, 5, padding=2)
        self.conv2 = torch.nn.ConvTranspose2d(64, 32, 5, padding=2)
        self.fc1 = torch.nn.Linear(latent_size, 64 * 28 * 28)

    def forward(self, x):
        x = torch.atanh(x)
        x = self.fc1(x)
        x = x.view(-1, 64, 28, 28)
        x = F.relu(self.conv2(x))
        x = self.conv1(x)
        return x


class TrainHopfield(torch.nn.Module):
    def __init__(self, label_size, latent_size):
        super(TrainHopfield, self).__init__()
        self.latent_size = latent_size
        self.label_size = label_size

    def forward(self, x):

        rho = torch.mean(x)

        for i, x_ in enumerate(x):
            temp = x_ - rho
            if i == 0:
                w = torch.ger(temp, temp)
            else:
                w += torch.ger(temp, temp)

        diag_w = torch.diag(torch.diag(w))
        w = w - diag_w
        w /= len(x)

        return w


class DeepHopfield(torch.nn.Module):
    def __init__(self, input_size, label_size, latent_size):
        super(DeepHopfield, self).__init__()
        self.label_size = label_size
        self.latent_size = latent_size
        self.input_size = input_size
        self.encoder = Encoder(latent_size).to(device)
        self.decoder = Decoder(latent_size).to(device)
        self.train_hopfield = TrainHopfield(label_size, latent_size)
        self.softmax = Softmax(latent_size, label_size).to(device)

        label_images = torch.rand((self.label_size, 1, 28, 28), device=device)
        self.label_latent_vectors = torch.tensor(self.encoder(label_images), requires_grad=True)

    def forward(self, images):
        input_latent_vectors = self.encoder(images)

        weight = self.train_hopfield(torch.sign(self.label_latent_vectors))
        copy_input_latent_vectors = torch.sign(input_latent_vectors).clone().detach()

        clustered_latent_vectors = torch.stack([
            self.clustering(copy_input_latent_vector, weight)
            for copy_input_latent_vector in copy_input_latent_vectors
        ])

        return (F.softmax(clustered_latent_vectors @ self.label_latent_vectors.T, dim=1),
                self.softmax(input_latent_vectors), self.decoder(input_latent_vectors))

    def clustering(self, init_s, w):
        s = init_s
        prev_e = self.energy(s, w)

        for _ in range(self.latent_size):
            prev_s = s.clone().detach()
            s = torch.sign(w @ prev_s)

            e = self.energy(s, w)

            if e == prev_e:
                return s

            prev_e = e
        return s

    @staticmethod
    def energy(s, w):
        return - s @ w @ s

    def optimizers(self):
        encoder_optimizer = \
            torch.optim.Adam([
                {'params': self.encoder.parameters()},
                {'params': self.softmax.parameters()},
                {'params': self.decoder.parameters()}
            ], lr=0.0001)

        prediction_optimizer = \
            torch.optim.Adam([
                self.label_latent_vectors
            ], lr=0.01)

        return encoder_optimizer, prediction_optimizer


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from PIL import Image
    import numpy as np
    import os

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()), batch_size=512, shuffle=True
    )

    model = DeepHopfield(784, 10, 16)
    opt1, opt2 = model.optimizers()

    klv_loss = torch.nn.KLDivLoss(reduction='batchmean')
    mse_lose = torch.nn.MSELoss()

    for step in range(10000):
        for b, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            opt1.zero_grad()
            opt2.zero_grad()

            clustered_label, label, restored_image = model(data)
            loss = klv_loss(clustered_label, label)
            image_loss = mse_lose(data, restored_image)
            loss = loss + image_loss

            loss.backward(retain_graph=True)
            image_loss.backward()

            print(f"[Epoch: {step: 05d} | {b: 02d} / {len(train_loader) - 1: 02d}] "
                  f"loss: {loss.data: .5f}", end='\r')
            opt1.step()
            opt2.step()

        print("")
        if step % 10 == 0:
            images = 255 * model.decoder(model.label_latent_vectors)
            os.makedirs(f'{step: 05d}', exist_ok=True)
            for i, image in enumerate(images):
                image = np.reshape(image, (28, 28))
                img = Image.fromarray(image)
                img = img.convert('L')
                img.save(f'{step: 05d}/{i}.jpg')

