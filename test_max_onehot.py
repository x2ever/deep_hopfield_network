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


xs = 2 * torch.rand((2, 2)) - 1
xs = torch.tensor(xs, requires_grad=True)
ys = torch.tensor([
    [-1.0, -1.0],
    [-1.0, +1.0],
])

zs = torch.tensor([
    0,
    1,
])

if __name__ == "__main__":
    my_max_onehot = MaxOneHot.apply

    optimizer = torch.optim.Adam([
        {'params': xs},
    ], lr=0.001)

    cross_entropy_loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(1000):
        optimizer.zero_grad()
        labels = torch.stack([
            my_max_onehot(torch.abs(x @ ys.T) / 2)
            for x in xs.clamp(-1, 1)
        ])
        loss = cross_entropy_loss_fn(labels, zs)
        loss.backward()
        optimizer.step()
        print(loss)

    print(loss, xs.clamp(-1, 1))