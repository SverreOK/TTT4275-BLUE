import numpy as np

class BLUE:
    def __init__(self, x, cov):
        n = x.size
        assert cov.shape == (n, n)
        cov_inv = np.linalg.inv(cov)
        U = np.ones((n,1))
        self.alpha = cov_inv @ U / (U.T @ cov_inv @ U)
        self.mean = (x @ self.alpha).item()
        self.sigma = np.sqrt(self.alpha.T @ cov @ self.alpha).item()
        delta = x - self.mean

    def get_blue(self):
        
        return x_hat