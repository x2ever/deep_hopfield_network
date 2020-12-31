import torch


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


xs = 2 * torch.rand((4, 2)) - 1
xs = torch.tensor(xs, requires_grad=True)
ys = torch.tensor([
    [-1.0, -1.0],
    [-1.0, +1.0],
    [+1.0, -1.0],
    [+1.0, +1.0]
])


xs_buffer = xs.clone().detach()
print(xs)

if __name__ == "__main__":
    my_sign = Sign.apply

    optimizer = torch.optim.Adam([
        {'params': xs},
    ], lr=0.001)

    mse_loss_fn = torch.nn.MSELoss()
    for _ in range(1000):
        optimizer.zero_grad()

        bi_xs = my_sign(xs.clamp(-1, 1))
        loss = mse_loss_fn(bi_xs, ys)
        loss.backward()
        print(loss, xs)
        # print((xs.data == xs_buffer.data).all())
        # xs.data = xs_buffer.data

        optimizer.step()

