import torch


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
        return grad_input


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


my_sign = Sign.apply
my_max_onehot = MaxOneHot.apply

xs = torch.tensor([
    [+1.0, +1.0, +1.0, +1.0,
     +1.0, +1.0, +1.0, +1.0,
     -1.0, -1.0, -1.0, -1.0,
     -1.0, -1.0, -1.0, -1.0],  # 0
    [+1.0, +1.0, +1.0, +1.0,
     -1.0, -1.0, -1.0, -1.0,
     +1.0, +1.0, +1.0, +1.0,
     -1.0, -1.0, -1.0, -1.0],  # 1
])
ys = torch.tensor([
    0,
    1,
])

target_tensors = torch.tensor(2 * torch.rand((2, 16)) - 1, requires_grad=True)


def get_weight(target_tensors_):

    rho = torch.mean(target_tensors_)

    for i, target_tensor_ in enumerate(target_tensors_):
        temp = target_tensor_ - rho
        if i == 0:
            w = torch.ger(temp, temp)
        else:
            w += torch.ger(temp, temp)

    diag_w = torch.diag(torch.diag(w))
    w = w - diag_w
    w /= len(target_tensors_)
    return w


def get_energy(w, s):
    return - s @ w @ s


def clustering(w, s):

    # min_s = my_sign(w @ s)
    # min_e = get_energy(w, s)
    #
    # for i in range(16):
    #     s = my_sign(w @ s)
    #     e = get_energy(w, s)
    #     if e < min_e:
    #         min_e = e
    #         min_s = s
    # return min_s

    e = get_energy(w, s)

    for _ in range(1):
        for i in range(16):
            s[i] = my_sign(w[i].T @ s.clone().detach())

        # new_e = get_energy(w, s)
        # if new_e == e:
        return s

        # e = new_e


optimizer = torch.optim.Adam([
        {'params': target_tensors},
    ], lr=0.001)
cross_entropy_loss_fn = torch.nn.CrossEntropyLoss()
mse_loss_fn = torch.nn.MSELoss()

for i in range(10000):

    optimizer.zero_grad()

    signed_target_tensors = my_sign(target_tensors.clamp(-1, 1))
    weight = get_weight(signed_target_tensors)
    clustered_xs = torch.stack([
        clustering(weight, x)
        for x in xs.clone()
    ])

    labels = torch.stack([
        my_max_onehot(torch.abs(clustered_x @ signed_target_tensors.T) / 16)
        for clustered_x in clustered_xs
    ])

    ts = signed_target_tensors[ys]

    cross_entropy_loss = cross_entropy_loss_fn(labels, ys)
    mse_loss = mse_loss_fn(ts, xs)
    loss = cross_entropy_loss + mse_loss
    loss.backward()
    print(i, loss, cross_entropy_loss, mse_loss, target_tensors)
    optimizer.step()

print(loss, signed_target_tensors, labels)