import os
import numpy as np
import Leverage

class Karmed(object):

    def __init__(self, arms, mus, sigmas):
        '''
        Define the number of Leverages you want, their mus (numpy array) and their sigmas (same)
        '''
        i = 0
        self.estimated_rewards = np.zeros(arms)
        self.no_selections = np.zeros(arms)
        self.leverages = []
        for arm in range(arms):
            self.leverages.append(Leverage.Leverage(mus[i], sigmas[i]))
            i += 1

        self.iter_per_run = 0
        self.epsilon = 0
        self.iter_values = []
        


    def set_train_params(self, iter_per_run, epsilon):
        '''
        Define number of training runs (epochs) and number of iterations for run
        '''
        self.iter_per_run = iter_per_run
        self.epsilon = epsilon
        self.iter_values = np.zeros(iter_per_run)

    
    def train(self):
        for iteration in range(self.iter_per_run):
            if (np.random.uniform(0, 1) <= self.epsilon):
                # Exploring
                index = np.random.randint(0, self.estimated_rewards.shape[0])
            else:
                # Exploiting
                temp = max(self.estimated_rewards)
                index = [i for i, j in enumerate(self.estimated_rewards) if j == temp]
                if len(index) > 1:
                    index = index[np.random.randint(0, len(index))]
                else:
                    index = index[0]
            
            self.no_selections[index] += 1
            self.iter_values[iteration] = self.leverages[index].pull()
            self.estimated_rewards[index] += 1/self.no_selections[index]*(self.iter_values[iteration] - self.estimated_rewards[index])
        
        return self.iter_values, self.estimated_rewards
            

                        




