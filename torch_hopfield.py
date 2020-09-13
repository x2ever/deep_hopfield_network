import numpy as np
import torch

class HopfieldNetwork(object):

    def __init__(self, size):
        self.size = size
        self.w = None
        self.train_data = None

    def train(self, train_data):
        self.train_data = train_data

        w = torch.zeros((self.size, self.size), requires_grad=False)
        rho = torch.mean(train_data)

        for td in train_data:
            t = td - rho
            w += torch.ger(t, t)

        # Make diagonal element of W into 0
        diag_w = torch.diag(torch.diag(w))
        w = w - diag_w
        w /= train_data.size()[0]

        self.w = w

    def predict(self, data):
        # Copy to avoid call by reference

        # Define predict list
        temp = []
        for _data in data.clone().detach():
            temp.append(
                (self.train_data == self._run(_data)).all(axis=1).int()
            )

        return torch.stack(temp)

    def _run(self, init_s):
        """
        Asynchronous update
        """
        # Compute initial state energy
        s = init_s
        e = self.energy(s)

        # Iteration
        while True:
            for j in range(self.size):
                # Update s
                temp = np.sign(self.w[j].T @ s)
                s[j] = torch.sign(self.w[j].T @ s)
            # # Compute new state energy
            e_new = self.energy(s)

            if e == e_new:
                return s
            e = e_new

    def energy(self, s):
        # temp = [torch.dot(s, w_) for w_ in self.w]
        # temp = torch.tensor(temp)
        return -0.5 * s @ self.w @ s


if __name__ == "__main__":
    H = HopfieldNetwork(size=16)

    train_data = torch.tensor(
        [
            [+1, -1, -1, -1, +1, -1, -1, -1, +1, -1, -1, -1, +1, +1, +1, +1],
            [+1, +1, +1, +1, +1, -1, -1, +1, +1, -1, -1, +1, +1, +1, +1, +1]
        ], dtype=torch.float32
    )

    test_data = torch.tensor(
        [
            [+1, -1, -1, -1, +1, -1, -1, -1, +1, -1, -1, -1, +1, +1, +1, -1],
            [+1, +1, +1, +1, +1, -1, -1, +1, +1, -1, -1, +1, +1, +1, +1, +1]
        ], dtype=torch.float32
    )

    H.train(train_data)
    print(H.predict(test_data))
