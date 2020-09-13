import numpy as np


class HopfieldNetwork(object):

    def __init__(self, size):
        self.size = size
        self.w = None

    def train(self, train_data):
        num_data = len(train_data)

        # initialize weights
        w = np.zeros((self.size, self.size))
        rho = np.sum([np.sum(t) for t in train_data]) / (num_data * self.size)
        print(rho)
        # Hebb rule
        for i in range(num_data):
            t = train_data[i] - rho
            w += np.outer(t, t)

        w /= num_data

        # Make diagonal element of W into 0
        diag_w = np.diag(np.diag(w))
        w = w - diag_w

        self.w = w

    def predict(self, data, num_iter=20, threshold=0, asyn=False):
        # Copy to avoid call by reference 
        copied_data = np.copy(data)

        # Define predict list
        predicted = []
        for i in range(len(data)):
            predicted.append(self._run(copied_data[i], num_iter=num_iter, threshold=threshold))
        return predicted

    def _run(self, init_s, num_iter, threshold):
        """
        Asynchronous update
        """
        # Compute initial state energy
        s = init_s
        e = self.energy(s, threshold=threshold)

        # Iteration
        for i in range(num_iter):
            for j in range(self.size):
                # Update s
                print(np.sign(self.w[j].T @ s - threshold) == s[j])
                s[j] = np.sign(self.w[j].T @ s - threshold)

            # Compute new state energy
            e_new = self.energy(s, threshold=threshold)
            print(e, e_new)

            # s is converged
            if e == e_new:
                return s
            # Update energy
            e = e_new
        return s

    def energy(self, s, threshold):
        return -0.5 * s @ self.w @ s + np.sum(s * threshold)


if __name__ == "__main__":
    H = HopfieldNetwork(size=16)

    train_data = np.array(
        [
            [+1, -1, -1, -1, +1, -1, -1, -1, +1, -1, -1, -1, +1, +1, +1, +1],
            [+1, +1, +1, +1, +1, -1, -1, +1, +1, -1, -1, +1, +1, +1, +1, +1]
        ], dtype=np.float32
    )

    test_data = np.array(
        [
            [+1, -1, -1, -1, +1, -1, -1, -1, +1, -1, -1, -1, +1, +1, +1, -1],
            [+1, +1, +1, +1, +1, -1, -1, +1, +1, -1, -1, +1, +1, +1, +1, +1]
        ], dtype=np.float32
    )

    H.train(train_data)
    predicted = H.predict(test_data)
