import numpy as np

class Leverage(object):

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    
    def pull(self):
        return np.random.normal(self.mu, self.sigma)


    def getMu(self):
        return self.mu


    def setMu(self, mu):
        self.mu = mu

    
    def getSigma(self):
        return self.sigma


    def setSigma(self, sigma):
        self.sigma = sigma

