import torch
import torch.nn.functional as F


device = torch.device("cuda:1")
latent_size = 48
label_size = 10
batch_size = 1000
total_epoch = 30


class MaxOneHot(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        idx = torch.argmax(input, dim=0)
        ctx._input_shape = input.shape
        ctx._input_dtype = input.dtype
        ctx._input_device = input.device
        ctx.save_for_backward(idx)
        output = torch.zeros(ctx._input_shape, device=ctx._input_device, dtype=ctx._input_dtype)
        output[idx] = input[idx]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        idx, = ctx.saved_tensors
        grad_input = torch.zeros(ctx._input_shape, device=ctx._input_device, dtype=ctx._input_dtype)
        grad_input[idx] = grad_output[idx]
        return 0.5 * grad_input


class Sign(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):
        ctx.save_for_backward(tensor)
        tensor = tensor.ge(0).float() * 2 - 1
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # tensor, = ctx.saved_tensors
        # grad_input = grad_output.clone()
        # grad_input = (tensor.lt(-1.0) + tensor.gt(1.0)).float().lt(0.5).float()*grad_input
        return grad_output


# Custom Layer
my_sign = Sign.apply
my_max_onehot = MaxOneHot.apply

# Loss Functions
cross_entropy_loss_fn = torch.nn.CrossEntropyLoss().to(device)
mse_loss_fn = torch.nn.MSELoss().to(device)
kl_div_loss_fn = torch.nn.KLDivLoss().to(device)


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

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, padding=1),  # batch x 16 x 28 x 28
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 32, 3, padding=1),  # batch x 32 x 28 x 28
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 64, 3, padding=1),  # batch x 32 x 28 x 28
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(2, 2)  # batch x 64 x 14 x 14
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, padding=1),  # batch x 64 x 14 x 14
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(128, 256, 3, padding=1),  # batch x 64 x 7 x 7
            torch.nn.ELU()
        )
        self.fc1 = torch.nn.Linear(256 * 7 * 7, latent_size)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(-1, 256 * 7 * 7)   # reshape Variable
        out = self.fc1(out)

        return torch.tanh(out)


class Decoder(torch.nn.Module):
    def __init__(self, latent_size):
        super(Decoder, self).__init__()
        self.latent_size = latent_size

        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),  # batch x 128 x 14 x 14
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.ConvTranspose2d(128, 64, 3, 1, 1),  # batch x 64 x 14 x 14
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(64)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 16, 3, 1, 1),  # batch x 16 x 14 x 14
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.ConvTranspose2d(16, 1, 3, 2, 1, 1),  # batch x 1 x 28 x 28
            torch.nn.ELU()
        )

        self.fc1 = torch.nn.Linear(latent_size, 256 * 7 * 7)

    def forward(self, x):
        out = self.fc1(x)
        out = out.view(-1, 256, 7, 7)
        out = self.layer1(out)
        out = self.layer2(out)
        return out


class Hopfield(torch.nn.Module):
    def __init__(self, latent_size, label_size, clustering_n=3):
        super(Hopfield, self).__init__()
        self.latent_size = latent_size
        self.label_size = label_size
        self.clustering_n = clustering_n

        label_latent_vector = 2 * torch.rand((label_size, latent_size), device=device) - 1
        self.label_latent_vectors = label_latent_vector.requires_grad_(True)

    def forward(self, s):
        # init_s = s.clone().detach()
        weight = self._get_weight()

        for _ in range(self.clustering_n):
            s = my_sign(weight @ s.clone())

        return torch.abs(
            s.clone() @ my_sign(self.label_latent_vectors.clamp(-1, 1)).T / self.latent_size
        )

    @staticmethod
    def _get_energy(s, w):
        return - s @ w @ s

    def _get_weight(self):

        x = my_sign(self.label_latent_vectors.clamp(-1, 1))
        rho = torch.mean(x)

        for i, x_ in enumerate(x):
            temp = x_ - rho
            if i == 0:
                weight = torch.ger(temp, temp)
            else:
                weight += torch.ger(temp, temp)

        diag_weight = torch.diag(torch.diag(weight))
        weight = weight - diag_weight
        weight /= len(x)

        return weight


class DeepHopfield:
    """
                  label latent vector ─(Hopfield)─> hopfield weight
                                ├────> [MSE Loss]        │
    input image ─(Encoder)─> latent vector ──────────────┴───> clustered label ──┬──> [KL-divergence]
         │                      ├──────(Softmax)─────────────> predicted label ──┘
         │                      └──────(Decoder)─> restored Image
         └────────────────────────────────────────────┴──────> [MSE Loss]
    """
    def __init__(self, latent_size, label_size):
        super(DeepHopfield, self).__init__()

        self.label_size = label_size
        self.latent_size = latent_size

        self.encoder = Encoder(latent_size).to(device)
        self.decoder = Decoder(latent_size).to(device)
        # self.softmax = Softmax(latent_size, label_size).to(device)

        self.hopfield = Hopfield(latent_size, label_size)

        self.hopfield_loss = None
        self.label_loss = None
        self.restoring_loss = None

    def forward(self, x, y):
        latent_vectors = self.encoder(x)
        bi_latent_vectors = my_sign(latent_vectors)

        # predicted_labels = self.softmax(bi_latent_vectors)
        clustered_labels_dist = torch.stack([
            self.hopfield(latent_vector)
            for latent_vector in bi_latent_vectors.clone()
        ])

        # clustered_labels = torch.argmax(clustered_labels_one_hot, dim=1)
        restored_image = self.decoder(bi_latent_vectors)

        self.hopfield_loss = mse_loss_fn(
            bi_latent_vectors, my_sign(self.hopfield.label_latent_vectors[y])
        )
        self.label_loss = \
            cross_entropy_loss_fn(clustered_labels_dist, y)
        # cross_entropy_loss_fn(predicted_labels, clustered_labels)

        self.restoring_loss = mse_loss_fn(x, restored_image)

        return torch.argmax(clustered_labels_dist, dim=1)

    def optimizer(self):
        opt = \
            torch.optim.Adam([
                {'params': self.encoder.parameters()},
                {'params': self.decoder.parameters()},
                # {'params': self.softmax.parameters()},
                {'params': self.hopfield.label_latent_vectors}
            ], lr=0.001)
        return opt


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from PIL import Image
    import numpy as np

    import os

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('.', train=True, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('.', train=False, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True
    )

    model = DeepHopfield(latent_size, label_size)
    optimizer = model.optimizer()

    print("Start Learning")
    for step in range(total_epoch + 1):
        print("")
        for b, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            clustered_labels = model.forward(data, target)
            accuracy = torch.sum(clustered_labels == target) / (len(target) / 100.0)
            hopfield_loss = model.hopfield_loss
            label_loss = model.label_loss
            restoring_loss = model.restoring_loss

            loss = hopfield_loss + label_loss + restoring_loss
            loss.backward()

            print(f"[Epoch: {step: 04d} | {b: 02d} / {len(train_loader) - 1: 02d}] "
                  f"loss: {loss.data: .3f} hopfield loss: {hopfield_loss.data: .3f} "
                  f"label loss: {label_loss.data: .3f} restoring loss: {restoring_loss.data: .3f} "
                  f"accuracy: {accuracy: .2f}\t", end='\r')
            optimizer.step()

        with torch.no_grad():
            print("")

            temp = 0
            for b, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                clustered_labels = model.forward(data, target)
                accuracy = torch.sum(clustered_labels == target) / (len(target) / 100.0)
                temp += accuracy / len(test_loader)
            print(f"[Epoch: {step: 04d}] "
                  f"accuracy: {temp: .2f}\t", end='\r')

            if step % 5 == 0:
                os.makedirs(f"{step}", exist_ok=True)
                for i, label_latent_vector in enumerate(model.hopfield.label_latent_vectors.clone().detach()):
                    image = model.decoder(my_sign(label_latent_vector)).clamp(0, 1).cpu().detach().numpy()
                    im = Image.fromarray((255 * image[0][0]).astype(np.uint8), 'L')
                    im.save(f"{step}/{i}.jpg")
