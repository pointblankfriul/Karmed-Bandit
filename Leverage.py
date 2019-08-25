import numpy as np

class Leverage(object):

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    
    def pull(self):
        return np.random.normal(self.mu, self.sigma)