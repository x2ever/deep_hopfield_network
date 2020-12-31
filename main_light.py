import torch
import torch.nn.functional as F


device = torch.device("cuda:0")
latent_size = 24
label_size = 10


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
kl_div_loss_fn = torch.nn.KLDivLoss(reduction='batchmean').to(device)
mse_loss_fn = torch.nn.MSELoss().to(device)


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
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
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
        x = F.elu(self.fc1(x))
        x = x.view(-1, 64, 28, 28)
        x = F.elu(self.conv2(x))
        x = self.conv1(x)
        return x


class Hopfield(torch.nn.Module):
    def __init__(self, latent_size, label_size, clustering_n=1):
        super(Hopfield, self).__init__()
        self.latent_size = latent_size
        self.label_size = label_size
        self.clustering_n = clustering_n

        label_latent_vector = 2 * torch.rand((label_size, latent_size), device=device) - 1
        self.label_latent_vectors = label_latent_vector.requires_grad_(True)

    def forward(self, s):
        weight = self._get_weight()

        for _ in range(self.clustering_n):
            for i in range(self.latent_size):
                s[i] = my_sign(weight[i].T @ s.clone().detach())

        return my_max_onehot(
            torch.abs(
                s.clone() @ my_sign(self.label_latent_vectors.clamp(-1, 1)).T
            ) / self.latent_size
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
    input image ─(Encoder)─> latent vector ──────────────┴───> clustered label ──┬──> [KL-div Loss]
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
        self.softmax = Softmax(latent_size, label_size).to(device)

        self.hopfield = Hopfield(latent_size, label_size)

        self.hopfield_loss = None
        self.label_loss = None
        self.restoring_loss = None

    def forward(self, x):
        latent_vectors = self.encoder(x)

        predicted_labels = self.softmax(latent_vectors)
        clustered_labels_one_hot = torch.stack([
            self.hopfield(latent_vector)
            for latent_vector in my_sign(latent_vectors).clone()
        ])
        clustered_labels = torch.argmax(clustered_labels_one_hot, dim=1)
        restored_image = self.decoder(latent_vectors)

        self.hopfield_loss = mse_loss_fn(latent_vectors, self.hopfield.label_latent_vectors[clustered_labels])
        self.label_loss = \
            0.5 * kl_div_loss_fn(predicted_labels, clustered_labels_one_hot) + \
            0.5 * kl_div_loss_fn(clustered_labels_one_hot, predicted_labels)
        self.restoring_loss = mse_loss_fn(x, restored_image)

        return clustered_labels

    def optimizer(self):
        opt = \
            torch.optim.Adam([
                {'params': self.encoder.parameters()},
                {'params': self.decoder.parameters()},
                {'params': self.softmax.parameters()},
                {'params': self.hopfield.label_latent_vectors}
            ], lr=0.001)
        return opt


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from PIL import Image

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('.', train=True, download=True, transform=transforms.ToTensor()), batch_size=4096, shuffle=True
    )

    model = DeepHopfield(latent_size, label_size)
    optimizer = model.optimizer()

    print("Start Learning")
    for step in range(1000):
        if step % 10 == 0:
            for label_latent_vector in model.hopfield.label_latent_vectors.clone().detach():
                image = model.decoder(label_latent_vector).clamp(-1, 1).cpu().numpy()
                print(image)

        for b, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            clustered_labels = model.forward(data)
            hopfield_loss = model.hopfield_loss
            label_loss = model.label_loss
            restoring_loss = model.restoring_loss

            loss = hopfield_loss + label_loss + restoring_loss
            loss.backward()

            print(f"[Epoch: {step: 04d} | {b: 02d} / {len(train_loader) - 1: 02d}] "
                  f"loss: {loss.data: .3f} hopfield loss: {hopfield_loss.data: .3f} "
                  f"label loss: {label_loss.data: .3f} restoring loss: {restoring_loss.data: .3f}", end='\r')
            optimizer.step()

        print("")
